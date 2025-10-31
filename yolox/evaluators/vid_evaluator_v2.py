#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import contextlib
import copy
import io
import itertools
import json
import tempfile
import time
import os
from loguru import logger
from tqdm import tqdm
from yolox.evaluators.coco_evaluator import per_class_AR_table, per_class_AP_table
import torch
import pycocotools.coco

from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)

vid_classes = (
    'airplane', 'antelope', 'bear', 'bicycle',
    'bird', 'bus', 'car', 'cattle',
    'dog', 'domestic_cat', 'elephant', 'fox',
    'giant_panda', 'hamster', 'horse', 'lion',
    'lizard', 'monkey', 'motorcycle', 'rabbit',
    'red_panda', 'sheep', 'snake', 'squirrel',
    'tiger', 'train', 'turtle', 'watercraft',
    'whale', 'zebra'
)

# from yolox.data.datasets.vid_classes import Arg_classes as  vid_classes

class VIDEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
            self, dataloader, img_size, confthre, nmsthre,
            num_classes, testdev=False, gl_mode=False,
            lframe=0, gframe=32, class_names=None, epoch=None, output_dir=None,
            enable_debug_outputs=False, **kwargs
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
            class_names (dict, optional): Dictionary mapping class index to class name.
                If None, uses default ImageNet VID class names.
            epoch (int, optional): Current training epoch for visualization naming.
            output_dir (str, optional): Directory to save visualization images.
            enable_debug_outputs (bool, optional): Enable validation batch visualization
                and prediction JSON accumulation. Defaults to False to avoid OOM.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.id = 0
        self.box_id = 0
        self.id_ori = 0
        self.box_id_ori = 0
        self.gl_mode = gl_mode
        self.lframe = lframe
        self.gframe = gframe
        self.kwargs = kwargs
        self.epoch = epoch
        self.output_dir = output_dir
        self.enable_debug_outputs = enable_debug_outputs
        # Use custom class_names if provided, otherwise use default ImageNet VID names
        if class_names is not None:
            self._class_names = class_names
        else:
            # Default to ImageNet VID classes
            self._class_names = {i: name for i, name in enumerate(vid_classes)}

        # Generate categories list from class_names
        categories = [{"supercategorie": "", "id": idx, "name": name}
                      for idx, name in sorted(self._class_names.items())]

        self.vid_to_coco = {
            'info': {
                'description': 'nothing',
            },
            'annotations': [],
            'categories': categories,
            'images': [],
            'licenses': []
        }
        self.vid_to_coco_ori = {
            'info': {
                'description': 'nothing',
            },
            'annotations': [],
            'categories': categories,
            'images': [],
            'licenses': []
        }
        self.testdev = testdev
        self.tmp_name_ori = './ori_pred.json'
        self.tmp_name_refined = './refined_pred.json'
        self.gt_ori = './gt_ori.json'
        self.gt_refined = './gt_refined.json'
        # File handle for streaming predictions (instead of storing in memory)
        self.predictions_file = None

    def evaluate(
            self,
            model,
            distributed=False,
            half=True,
            trt_file=None,
            decoder=None,
            test_size=None,
            img_path=None,
            epoch=None,
            output_dir=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """

        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        labels_list = []
        ori_data_list = []
        ori_label_list = []
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        # Open predictions file for streaming writes (JSONL format)
        if self.enable_debug_outputs and output_dir is not None and is_main_process():
            predictions_path = f"{output_dir}/predictions.jsonl"
            self.predictions_file = open(predictions_path, 'w')
            logger.info(f"Writing predictions to {predictions_path} (JSONL format)")

        try:
            for cur_iter, (imgs, _, info_imgs, label, path, time_embedding) in enumerate(
                progress_bar(self.dataloader)
            ):

                with torch.no_grad():
                    imgs = imgs.type(tensor_type)
                    # skip the the last iters since batchsize might be not enough for batch inference
                    is_time_record = cur_iter < len(self.dataloader) - 1
                    if is_time_record:
                        start = time.time()
                    outputs, ori_res, (o,f,c) = model(imgs,
                                             lframe=self.lframe,
                                             gframe = self.gframe,
                                             )

                    if is_time_record:
                        infer_end = time_synchronized()
                        inference_time += infer_end - start

                # Visualize first validation batch each epoch
                if self.enable_debug_outputs and cur_iter == 0 and output_dir is not None and is_main_process():
                    self.visualize_batch(imgs, label, path, outputs, output_dir, epoch)

                # Accumulate granular prediction data
                if self.enable_debug_outputs and output_dir is not None:
                    self.accumulate_predictions(
                        paths=path,
                        labels=label,
                        outputs=o,
                        fc_output=f,
                        conf_output=c
                    )

                if self.gl_mode:
                    local_num = int(imgs.shape[0] / 2)
                    info_imgs = info_imgs[:local_num]
                    label = label[:local_num]
                if self.kwargs.get('first_only',False):
                    info_imgs = [info_imgs[0]]
                    label = [label[0]]
                temp_data_list, temp_label_list = self.convert_to_coco_format(outputs, info_imgs, copy.deepcopy(label))
                data_list.extend(temp_data_list)
                labels_list.extend(temp_label_list)

            self.vid_to_coco['annotations'].extend(labels_list)
            statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
            if distributed:
                data_list = gather(data_list, dst=0)
                data_list = list(itertools.chain(*data_list))
                torch.distributed.reduce(statistics, dst=0)

            del labels_list
            eval_results = self.evaluate_prediction(data_list, statistics)
            del data_list
            self.vid_to_coco['annotations'] = []

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise
        finally:
            # Ensure predictions file is closed even if evaluation fails
            if self.predictions_file is not None:
                self.predictions_file.close()
                self.predictions_file = None
                if output_dir is not None and is_main_process():
                    logger.info(f"Finished writing predictions to {output_dir}/predictions.jsonl")

        synchronize()
        return eval_results

    def convert_to_coco_format(self, outputs, info_imgs, labels):
        data_list = []
        label_list = []
        frame_now = 0

        for (output, info_img, _label) in zip(
                outputs, info_imgs, labels
        ):
            # if frame_now>=self.lframe: break
            scale = min(
                self.img_size[0] / float(info_img[0]), self.img_size[1] / float(info_img[1])
            )
            bboxes_label = _label[:, 1:]
            bboxes_label /= scale
            bboxes_label = xyxy2xywh(bboxes_label)
            cls_label = _label[:, 0]
            for ind in range(bboxes_label.shape[0]):
                label_pred_data = {
                    "image_id": int(self.id),
                    "category_id": int(cls_label[ind]),
                    "bbox": bboxes_label[ind].numpy().tolist(),
                    "segmentation": [],
                    'id': self.box_id,
                    "iscrowd": 0,
                    'area': int(bboxes_label[ind][2] * bboxes_label[ind][3])
                }  # COCO json format
                self.box_id = self.box_id + 1
                label_list.append(label_pred_data)
            self.vid_to_coco['images'].append({'id': self.id})

            if output is None:
                self.id = self.id + 1
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]
            # preprocessing: resize
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            for ind in range(bboxes.shape[0]):
                label = int(cls[ind])
                pred_data = {
                    "image_id": int(self.id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
            self.id = self.id + 1
            frame_now = frame_now + 1

        return data_list, label_list

    def accumulate_predictions(self, paths, labels, outputs, fc_output, conf_output):
        """
        Write prediction data directly to JSONL file (one JSON object per line).
        This avoids memory accumulation for large validation sets.

        Args:
            paths: list of image paths
            labels: list of label tensors
            outputs: tensor of raw model outputs
            fc_output: tensor of FC layer outputs
            conf_output: tensor of confidence outputs
        """
        import torch

        # Only write if file is open
        if self.predictions_file is None:
            return

        # Convert tensors to JSON-serializable format
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.cpu()
        if isinstance(fc_output, torch.Tensor):
            fc_output = fc_output.cpu()
        if isinstance(conf_output, torch.Tensor):
            conf_output = conf_output.cpu()

        for i, path in enumerate(paths):
            # Convert label tensor to list
            if isinstance(labels[i], torch.Tensor):
                label_list = labels[i].cpu().tolist()
            else:
                label_list = labels[i].tolist() if hasattr(labels[i], 'tolist') else labels[i]

            output_slice = outputs[i]

            # Create prediction data for this image
            pred_data = {
                'path': path,
                'labels': label_list,
                'bbox_preds': output_slice[:, :4].tolist() if output_slice.shape[0] > 0 else [],
                'obj_preds': output_slice[:, 4].tolist() if output_slice.shape[0] > 0 and output_slice.shape[1] > 4 else [],
                'cls_preds': output_slice[:, 5:].tolist() if output_slice.shape[0] > 0 and output_slice.shape[1] > 5 else [],
                'fc_outputs': fc_output[i].tolist() if fc_output is not None else None,
                'conf_outputs': conf_output[i].tolist() if conf_output is not None else None
            }

            # Write as a single line (JSONL format)
            self.predictions_file.write(json.dumps(pred_data) + '\n')

    def write_predictions_json(self, filename):
        """
        DEPRECATED: Predictions are now written incrementally during evaluation.
        This method is kept for backwards compatibility but does nothing.

        Args:
            filename: path to output JSON file (ignored)
        """
        logger.warning(
            "write_predictions_json() is deprecated. "
            "Predictions are now written incrementally to JSONL format during evaluation."
        )


    def visualize_batch(self, imgs, labels, paths, outputs, output_dir, epoch):
        """
        Visualize first validation batch with ground truth and predictions.

        Args:
            imgs: [batch_size, C, H, W] tensor
            labels: list of [N_objects, 5] tensors (x1, y1, x2, y2, class) in pixel coords
            paths: list of image paths
            outputs: tuple of (result, result_ori) where result is list of [N_preds, 7+num_classes] tensors
            output_dir: directory to save visualization
            epoch: current epoch number
        """
        from yolox.utils import plot_images
        import numpy as np
        import torch

        # outputs is a tuple (result, result_ori), we want result
        predictions = outputs[0] if isinstance(outputs, tuple) else outputs

        batch_size = imgs.shape[0]

        # Convert targets from list of [N_objects, 5] to [batch_size, max_objs, 6] format
        # Input format: [x1, y1, x2, y2, class] in pixel coordinates
        # Output format needed: [batch_idx, class, x_center, y_center, w, h] (normalized 0-1)
        max_objs = max([label.shape[0] for label in labels]) if labels else 0
        if max_objs == 0:
            max_objs = 1  # Avoid empty tensor
        targets = np.zeros((batch_size, max_objs, 6))

        h, w = self.img_size
        for i, label in enumerate(labels):
            if isinstance(label, torch.Tensor):
                label = label.cpu().numpy()

            if label.shape[0] > 0:
                n_objs = label.shape[0]
                # label is [N, 5] with format [x1, y1, x2, y2, class]
                x1, y1, x2, y2 = label[:, 0], label[:, 1], label[:, 2], label[:, 3]
                cls = label[:, 4]

                # Filter out invalid boxes (negative coords, zero area, etc)
                valid = (x2 > x1) & (y2 > y1) & (x1 >= 0) & (y1 >= 0)
                if not valid.any():
                    continue

                x1, y1, x2, y2, cls = x1[valid], y1[valid], x2[valid], y2[valid], cls[valid]
                n_valid = valid.sum()

                # Convert xyxy to xywh (center format) and normalize
                xc = ((x1 + x2) / 2) / w
                yc = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                targets[i, :n_valid, 0] = i  # batch index
                targets[i, :n_valid, 1] = cls  # class
                targets[i, :n_valid, 2] = xc  # x_center (normalized)
                targets[i, :n_valid, 3] = yc  # y_center (normalized)
                targets[i, :n_valid, 4] = bw  # width (normalized)
                targets[i, :n_valid, 5] = bh  # height (normalized)

        # Convert predictions from list of [N_preds, 7+] to [batch_size, max_preds, 5] format
        # Input format: [x1, y1, x2, y2, obj_conf, class_conf, class, ...] in pixel coords
        # Output format needed: [class, x_center, y_center, w, h] (normalized 0-1)
        max_preds = max([pred.shape[0] if pred is not None else 0 for pred in predictions])
        if max_preds == 0:
            preds = None  # No predictions
        else:
            preds = np.zeros((batch_size, max_preds, 5))
            for i, pred in enumerate(predictions):
                if pred is not None and pred.shape[0] > 0:
                    if isinstance(pred, torch.Tensor):
                        pred = pred.cpu().numpy()

                    n_preds = pred.shape[0]
                    # pred is [N, 7+] with format [x1, y1, x2, y2, obj_conf, class_conf, class, ...]
                    x1, y1, x2, y2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
                    cls = pred[:, 6]

                    # Filter out invalid boxes
                    valid = (x2 > x1) & (y2 > y1) & (x1 >= 0) & (y1 >= 0)
                    if not valid.any():
                        continue

                    x1, y1, x2, y2, cls = x1[valid], y1[valid], x2[valid], y2[valid], cls[valid]
                    n_valid = valid.sum()

                    # Convert xyxy to xywh (center format) and normalize
                    xc = ((x1 + x2) / 2) / w
                    yc = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h

                    preds[i, :n_valid, 0] = cls  # class
                    preds[i, :n_valid, 1] = xc  # x_center (normalized)
                    preds[i, :n_valid, 2] = yc  # y_center (normalized)
                    preds[i, :n_valid, 3] = bw  # width (normalized)
                    preds[i, :n_valid, 4] = bh  # height (normalized)

        # Create filename
        if epoch is not None:
            fname = f"{output_dir}/val_batch_epoch_{epoch}.jpg"
        else:
            fname = f"{output_dir}/val_batch.jpg"

        # Call plot_images
        plot_images(imgs, targets, paths, fname=fname, preds=preds, names=self._class_names)
        logger.info(f"Saved validation batch visualization to {fname}")

    def convert_to_coco_format_ori(self, outputs, info_imgs, labels):

        data_list = []
        label_list = []
        frame_now = 0
        for (output, info_img, _label) in zip(
                outputs, info_imgs, labels
        ):
            scale = min(
                self.img_size[0] / float(info_img[0]), self.img_size[1] / float(info_img[1])
            )
            bboxes_label = _label[:, 1:]
            bboxes_label /= scale
            bboxes_label = xyxy2xywh(bboxes_label)
            cls_label = _label[:, 0]
            for ind in range(bboxes_label.shape[0]):
                label_pred_data = {
                    "image_id": int(self.id_ori),
                    "category_id": int(cls_label[ind]),
                    "bbox": bboxes_label[ind].numpy().tolist(),
                    "segmentation": [],
                    'id': self.box_id_ori,
                    "iscrowd": 0,
                    'area': int(bboxes_label[ind][2] * bboxes_label[ind][3])
                }  # COCO json format
                self.box_id_ori = self.box_id_ori + 1
                label_list.append(label_pred_data)

                # print('label:',label_pred_data)

            self.vid_to_coco_ori['images'].append({'id': self.id_ori})

            if output is None:
                self.id_ori = self.id_ori + 1
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            # print(cls.shape)
            for ind in range(bboxes.shape[0]):
                label = int(cls[ind])
                pred_data = {
                    "image_id": int(self.id_ori),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)

            self.id_ori = self.id_ori + 1
            frame_now = frame_now + 1
        return data_list, label_list

    def evaluate_prediction(self, data_dict, statistics, ori=False):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_sampler.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_sampler.batch_size)
        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                ["forward", "NMS", "inference"],
                [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
            )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:

            _, tmp = tempfile.mkstemp()
            if ori:
                json.dump(self.vid_to_coco_ori, open(self.gt_ori, 'w'))
                json.dump(data_dict, open(self.tmp_name_ori, 'w'))
                json.dump(self.vid_to_coco_ori, open(tmp, "w"))
            else:
                json.dump(self.vid_to_coco, open(self.gt_refined, 'w'))
                json.dump(data_dict, open(self.tmp_name_refined, 'w'))
                json.dump(self.vid_to_coco, open(tmp, "w"))

            cocoGt = pycocotools.coco.COCO(tmp)
            # TODO: since pycocotools can't process dict in py36, write data to json file.

            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, "w"))
            cocoDt = cocoGt.loadRes(tmp)
            try:
                from yolox.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools.cocoeval import COCOeval

                logger.warning("Use standard COCOeval.")

            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()

            cat_ids = list(cocoGt.cats.keys())
            cat_names = [cocoGt.cats[catId]['name'] for catId in sorted(cat_ids)]
            AP_table = per_class_AP_table(cocoEval, class_names=cat_names)
            info += "per class AP:\n" + AP_table + "\n"

            AR_table = per_class_AR_table(cocoEval, class_names=cat_names)
            info += "per class AR:\n" + AR_table + "\n"
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info
