#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import copy
import os
import random

import numpy
from loguru import logger

import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset as torchDataset
from torch.utils.data.sampler import Sampler,BatchSampler,SequentialSampler
from xml.dom import minidom
import math
from yolox.utils import xyxy2cxcywh

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png",".JPEG"]
XML_EXT = [".xml"]
name_list = ['n02691156','n02419796','n02131653','n02834778','n01503061','n02924116','n02958343','n02402425','n02084071','n02121808','n02503517','n02118333','n02510455','n02342885','n02374451','n02129165','n01674464','n02484322','n03790512','n02324045','n02509815','n02411705','n01726692','n02355227','n02129604','n04468005','n01662784','n04530566','n02062744','n02391049']
numlist = range(30)
name_num = dict(zip(name_list,numlist))

class VIDDataset(torchDataset):
    """
    VID sequence
    """

    def __init__(
        self,
        file_path="train_seq.npy",
        img_size=(416, 416),
        preproc=None,
        lframe = 18,
        gframe = 6,
        val = False,
        mode='random',
        dataset_pth = '',
        tnum = 1000,
        formal = False,
        traj_linking = False,
        local_stride = 1,
        class_id_map = None
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
            class_id_map (dict, optional): Mapping from XML class names to class indices.
                If None, uses default ImageNet VID mapping.
        """
        super().__init__()
        self.tnum = tnum
        self.traj_linking = traj_linking
        self.input_dim = img_size
        self.file_path = file_path
        self.mode = mode  # random, continous, uniform
        self.img_size = img_size
        self.preproc = preproc
        self.val = val
        self.formal = formal
        self.local_stride = local_stride
        self.res = self.photo_to_sequence(self.file_path,lframe,gframe)
        self.dataset_pth = dataset_pth
        # Use custom class_id_map if provided, otherwise use default ImageNet VID mapping
        self.class_id_map = class_id_map if class_id_map is not None else name_num

    def __len__(self):
        return len(self.res)


    def photo_to_sequence(self,dataset_path,lframe,gframe):
        '''

        Args:
            dataset_path: list,every element is a list contain all frames in a video dir
        Returns:
            split result
        '''
        res = []
        dataset = np.load(dataset_path,allow_pickle=True).tolist()
        for element in dataset: # element = one video sequence
            ele_len = len(element)
            if ele_len<lframe+gframe:
                #TODO fix the unsolved part
                if self.formal:
                    res.append(element)
                else:
                    continue
                # res.append(element)
                # continue
            else:
                if self.mode == 'random':
                    if lframe == 0:
                        split_num = int(ele_len / (gframe))
                        random.shuffle(element)
                        for i in range(split_num):
                            res.append(element[i * gframe:(i + 1) * gframe])
                        if self.formal and len(element[split_num * gframe:]):
                            tail = element[split_num * gframe:]
                            # padding = tail + element[:gframe-len(tail)]
                            res.append(tail)
                    elif lframe!=0:
                        if self.local_stride==1:
                            split_num = int(ele_len / (lframe))
                            all_local_frame = element[:split_num * lframe]
                            for i in range(split_num):
                                if self.traj_linking and i!=0:
                                    l_frame = all_local_frame[i * lframe-1:(i + 1) * lframe]
                                else:
                                    l_frame = all_local_frame[i * lframe:(i + 1) * lframe]
                                g_frame = random.sample(element[:i * lframe] + element[(i + 1) * lframe:], gframe)
                                res.append(l_frame + g_frame)
                            if self.formal and len(element[split_num * lframe:]):
                                if self.traj_linking:
                                    tail = element[split_num * lframe-1:]
                                else:
                                    tail = element[split_num * lframe:]
                                res.append(tail)
                        else:
                            split_num = ele_len//(lframe*self.local_stride)
                            for i in range(split_num):
                                for j in range(self.local_stride):
                                    res.append(element[lframe * self.local_stride * i:lframe * self.local_stride * (i + 1)][
                                                   j::self.local_stride])
                    else:
                        print('unsupport mode, exit')
                        exit(0)

                elif self.mode == 'uniform':
                    split_num = int(ele_len / (gframe))
                    all_uniform_frame = element[:split_num * gframe]
                    for i in range(split_num):
                        res.append(all_uniform_frame[i::split_num])

                else:
                    print('unsupport mode, exit')
                    exit(0)

        if self.val:
            if self.tnum == -1:
                return res
            else:
                return res[:self.tnum]
        else:
            random.shuffle(res)
            return res[:15000]


    def get_annotation(self,path,test_size):
        path = path.replace("Data","Annotations").replace("JPEG","xml").replace("jpg", "xml").replace("PNG", "xml").replace("png", "xml")
        if os.path.isdir(path):
            files = get_xml_list(path)
        else:
            files = [path]
        files.sort()
        anno_res = []
        for xmls in files:
            photoname = xmls.replace("Annotations","Data").replace("xml","JPEG")
            file = minidom.parse(xmls)
            root = file.documentElement
            objs = root.getElementsByTagName("object")
            width = int(root.getElementsByTagName('width')[0].firstChild.data)
            height = int(root.getElementsByTagName('height')[0].firstChild.data)
            tempnode = []
            for obj in objs:
                nameNode = obj.getElementsByTagName("name")[0].firstChild.data
                xmax = int(obj.getElementsByTagName("xmax")[0].firstChild.data)
                xmin = int(obj.getElementsByTagName("xmin")[0].firstChild.data)
                ymax = int(obj.getElementsByTagName("ymax")[0].firstChild.data)
                ymin = int(obj.getElementsByTagName("ymin")[0].firstChild.data)
                x1 = np.max((0,xmin))
                y1 = np.max((0,ymin))
                x2 = np.min((width,xmax))
                y2 = np.min((height,ymax))
                if x2 >= x1 and y2 >= y1:
                    #tempnode.append((self.class_id_map[nameNode],x1,y1,x2,y2,))
                    tempnode.append(( x1, y1, x2, y2,self.class_id_map[nameNode],))
            num_objs = len(tempnode)
            res = np.zeros((num_objs, 5))
            r = min(test_size[0] / height, test_size[1] / width)
            for ix, obj in enumerate(tempnode):
                res[ix, 0:5] = obj[0:5]
            res[:, :-1] *= r
            anno_res.append(res)
        return anno_res


    def pull_item(self,path):
        """
                One image / label pair for the given index is picked up and pre-processed.

                Args:
                    index (int): data index

                Returns:
                    img (numpy.ndarray): pre-processed image
                    padded_labels (torch.Tensor): pre-processed label data.
                        The shape is :math:`[max_labels, 5]`.
                        each label consists of [class, xc, yc, w, h]:
                            class (float): class index.
                            xc, yc (float) : center of bbox whose values range from 0 to 1.
                            w, h (float) : size of bbox whose values range from 0 to 1.
                    info_img : tuple of h, w.
                        h, w (int): original shape of the image
                    img_id (int): same as the input index. Used for evaluation.
                """
        path = os.path.join(self.dataset_pth,path)
        annos = self.get_annotation(path, self.img_size)[0]

        img = cv2.imread(path)
        height, width = img.shape[:2]
        img_info = (height, width)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return img, annos, img_info, path

    def __getitem__(self, path):
        img, target, img_info, path = self.pull_item(path)
        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info,path

class Arg_VID(torchDataset):
    """
    VID sequence
    """

    def __init__(
        self,
        data_dir='/media/tuf/ssd/Argoverse-1.1/',
        img_size=(416, 640),
        preproc=None,
        lframe = 0,
        gframe = 16,
        val = False,
        mode='random',
        COCO_anno = '',
        name = "tracking",
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__()
        self.input_dim = img_size
        self.name = name
        self.val = val
        self.data_dir = data_dir
        self.img_size = img_size
        self.coco_anno_path = COCO_anno
        self.name_id_dic = self.get_NameId_dic()
        self.coco = COCO(COCO_anno)
        remove_useless_info(self.coco)
        self.ids = sorted(self.coco.getImgIds())
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])
        self.annotations = self._load_coco_annotations()
        self.mode = mode  # random, continous, uniform
        self.preproc = preproc

        self.res = self.photo_to_sequence(lframe,gframe)

    def get_NameId_dic(self):
        img_dic = {}
        with open(self.coco_anno_path,'r') as train_anno_content:
            train_anno_content = json.load(train_anno_content)
            for im in train_anno_content['images']:
                img_dic[im['name']] = im['id']
        return img_dic

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def __len__(self):
        return len(self.res)

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        im_ann['name'] = self.coco.dataset['seq_dirs'][im_ann['sid']] + '/' + im_ann['name']
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 5))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = (
            im_ann["name"]
            if "name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        return (res, img_info, resized_info, file_name)

    def photo_to_sequence(self,lframe,gframe, seq_len = 192):
        '''

        Args:
            dataset_path: list,every element is a list contain all frame in a video dir
        Returns:
            split result
        '''
        res = []

        with open(self.coco_anno_path, 'r') as anno:
            anno = json.load(anno)
            dataset = [[] for i in range(len(anno['sequences']))]
            for im in anno['images']:
                dataset[im['sid']].append(self.coco.dataset['seq_dirs'][im['sid']] + '/' + im['name'])
            for ele in dataset:
                sorted(ele)

        for element in dataset:
            ele_len = len(element)
            if ele_len<lframe+gframe:
                #TODO fix the unsolved part
                #res.append(element)
                continue
            else:
                if self.mode == 'random':
                    # split_num = int(ele_len / (gframe))
                    # random.shuffle(element)
                    # for i in range(split_num):
                    #     res.append(element[i * gframe:(i + 1) * gframe])
                    # if self.val and element[(i + 1) * gframe:] != []:
                    #     res.append(element[(i + 1) * gframe:])

                    seq_split_num = int(len(element) / seq_len)
                    for k in range(seq_split_num + 1):
                        tmp = element[k * seq_len:(k + 1) * seq_len]
                        if tmp == []:continue
                        random.shuffle(tmp)
                        split_num = int(len(tmp) / (gframe))
                        for i in range(split_num):
                            res.append(tmp[i * gframe:(i + 1) * gframe])
                        if self.val and tmp[(i + 1) * gframe:] != []:
                            res.append(tmp[(i + 1) * gframe:])
                elif self.mode == 'uniform':
                    split_num = int(ele_len / (gframe))
                    all_uniform_frame = element[:split_num * gframe]
                    for i in range(split_num):
                        res.append(all_uniform_frame[i::split_num])
                elif self.mode == 'gl':
                    split_num = int(ele_len / (lframe))
                    all_local_frame = element[:split_num * lframe]
                    for i in range(split_num):
                        g_frame = random.sample(element[:i * lframe] + element[(i + 1) * lframe:], gframe)
                        res.append(all_local_frame[i * lframe:(i + 1) * lframe] + g_frame)
                else:
                    print('unsupport mode, exit')
                    exit(0)

        if self.val:
            # random.seed(42)
            # random.shuffle(res)
            return res#[:1000]#[1000:1250]#[2852:2865]
        else:
            random.shuffle(res)
            return res#[:1000]#[:15000]


    def pull_item(self,path):
        """
                One image / label pair for the given index is picked up and pre-processed.

                Args:
                    index (int): data index

                Returns:
                    img (numpy.ndarray): pre-processed image
                    padded_labels (torch.Tensor): pre-processed label data.
                        The shape is :math:`[max_labels, 5]`.
                        each label consists of [class, xc, yc, w, h]:
                            class (float): class index.
                            xc, yc (float) : center of bbox whose values range from 0 to 1.
                            w, h (float) : size of bbox whose values range from 0 to 1.
                    info_img : tuple of h, w.
                        h, w (int): original shape of the image
                    img_id (int): same as the input index. Used for evaluation.
                """
        path = path.split('/')[-1]
        idx = self.name_id_dic[path]
        annos, img_info, resized_info, img_path = self.annotations[idx]
        abs_path = os.path.join(self.data_dir, self.name, img_path)
        img = cv2.imread(abs_path)

        height, width = img.shape[:2]
        img_info = (height, width)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return img, annos.copy(), img_info, img_path

    def __getitem__(self, path):

        img, target, img_info, path = self.pull_item(path)
        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info,path


class OVIS(Arg_VID):
    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        #im_ann['name'] = self.coco.dataset['seq_dirs'][im_ann['sid']] + '/' + im_ann['name']
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 5))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = (
            im_ann["name"]
            if "name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        return (res, img_info, resized_info, file_name)

    def photo_to_sequence(self,lframe,gframe):
        '''

        Args:
            dataset_path: list,every element is a list contain all frame in a video dir
        Returns:
            split result
        '''
        res = []

        with open(self.coco_anno_path, 'r') as anno:
            anno = json.load(anno)
            dataset = [[] for i in range(len(anno['videos']))]
            for im in anno['images']:
                dataset[im['sid']].append(im['name'])
            for ele in dataset:
                sorted(ele)

        for element in dataset:
            ele_len = len(element)
            if ele_len<lframe+gframe:
                #TODO fix the unsolved part
                #res.append(element)
                continue
            else:
                if self.mode == 'random':
                    split_num = int(ele_len / (gframe))
                    random.shuffle(element)
                    for i in range(split_num):
                        res.append(element[i * gframe:(i + 1) * gframe])
                elif self.mode == 'uniform':
                    split_num = int(ele_len / (gframe))
                    all_uniform_frame = element[:split_num * gframe]
                    for i in range(split_num):
                        res.append(all_uniform_frame[i::split_num])
                elif self.mode == 'gl':
                    split_num = int(ele_len / (lframe))
                    all_local_frame = element[:split_num * lframe]
                    for i in range(split_num):
                        g_frame = random.sample(element[:i * lframe] + element[(i + 1) * lframe:], gframe)
                        res.append(all_local_frame[i * lframe:(i + 1) * lframe] + g_frame)
                else:
                    print('unsupport mode, exit')
                    exit(0)

        if self.val:
            random.seed(42)
            random.shuffle(res)
            return res#[2000:3000]#[1000:1250]#[2852:2865]
        else:
            random.shuffle(res)
            return res#[:15000]

    def pull_item(self,path):
        """
                One image / label pair for the given index is picked up and pre-processed.

                Args:
                    index (int): data index

                Returns:
                    img (numpy.ndarray): pre-processed image
                    padded_labels (torch.Tensor): pre-processed label data.
                        The shape is :math:`[max_labels, 5]`.
                        each label consists of [class, xc, yc, w, h]:
                            class (float): class index.
                            xc, yc (float) : center of bbox whose values range from 0 to 1.
                            w, h (float) : size of bbox whose values range from 0 to 1.
                    info_img : tuple of h, w.
                        h, w (int): original shape of the image
                    img_id (int): same as the input index. Used for evaluation.
                """
        idx = self.name_id_dic[path]
        annos, img_info, resized_info, img_path = self.annotations[idx]
        abs_path = os.path.join(self.data_dir,self.name, img_path)
        img = cv2.imread(abs_path)

        height, width = img.shape[:2]
        img_info = (height, width)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return img, annos.copy(), img_info, img_path



def get_xml_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in XML_EXT:
                image_names.append(apath)

    return image_names

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

def make_path(train_dir,save_path):
    res = []
    for root,dirs,files in os.walk(train_dir):
        temp = []
        for filename in files:
            apath = os.path.join(root, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                temp.append(apath)
        if(len(temp)):
            temp.sort()
            res.append(temp)
    res_np = np.array(res,dtype=object)
    np.save(save_path,res_np)


class TestSampler(SequentialSampler):
    def __init__(self,data_source):
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self):
        return iter(self.data_source.res)

    def __len__(self):
        return len(self.data_source)

class TrainSampler(Sampler):
    def __init__(self,data_source):
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self):
        random.shuffle(self.data_source.res)
        return iter(self.data_source.res)

    def __len__(self):
        return len(self.data_source)

class VIDBatchSampler(BatchSampler):
    def __iter__(self):
        batch = []
        for ele in self.sampler:
            for filename in ele:
                batch.append(filename)
                if (len(batch)) == self.batch_size:
                    yield batch
                    batch = []
        if len(batch)>0 and not self.drop_last:
            yield batch
    def __len__(self):
        return len(self.sampler)

class VIDBatchSampler_Test(BatchSampler):
    def __iter__(self):
        batch = []
        for ele in self.sampler:
            yield ele
            # for filename in ele:
            #     batch.append(filename)
            #     if (len(batch)) == self.batch_size:
            #         yield batch
            #         batch = []
            # if len(batch)>0 and not self.drop_last:
            #     yield batch
    def __len__(self):
        return len(self.sampler)

class OnePerBatchSampler(Sampler):
    def __init__(self, file_path, batch_size, shuffle=True):
        self.filepath = file_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.arr = np.load(file_path, allow_pickle=True)
        self.indices = self.get_starting_file_indices()
    def get_starting_file_indices(self):
        # return list containing tuples of all indices greater than batch_size away from the end of the video
        indices = [(idx1, idx2) for idx1 in range(len(self.arr)) for idx2 in range(len(self.arr[idx1]) - self.batch_size)]
        return indices
    def __iter__(self):
        epoch_indices = self.indices.copy()
        if self.shuffle:
            epoch_indices = random.sample(epoch_indices, len(self))
        else:
            epoch_indices = epoch_indices[:len(self)]
        return iter(epoch_indices)
    def __len__(self): # how many calls to __iter__ per epoch = number of batches (each call produces one starting idx = one batch)
        num_files = 0
        for ele in self.arr:
            num_files += len(ele)
        return num_files // self.batch_size

class OnePerBatchSampler_Val(Sampler):
    """
    Validation sampler for pred_mode that creates overlapping batches.
    Each batch overlaps with the next by 1 frame (stride = batch_size - 1).
    The final batch always starts at (video_length - batch_size) to ensure
    the entire video (except first frame) gets predictions.
    """
    def __init__(self, file_path, batch_size):
        self.filepath = file_path
        self.batch_size = batch_size
        self.arr = np.load(file_path, allow_pickle=True)
        self.indices = self.get_overlapping_indices()

    def get_overlapping_indices(self):
        """
        Generate overlapping batch indices where each batch overlaps by 1 frame.
        For a video of length L and batch_size B:
        - Start at index 0
        - Stride by (B-1) each time
        - Always end with a final batch at index (L-B)

        Example: video_length=10, batch_size=4 -> indices [0, 3, 6]
        Example: video_length=5, batch_size=4 -> indices [0, 1]
        """
        indices = []
        stride = self.batch_size - 1

        for video_idx in range(len(self.arr)):
            video_length = len(self.arr[video_idx])

            # Skip videos shorter than batch_size
            if video_length < self.batch_size:
                continue

            final_idx = video_length - self.batch_size
            idx = 0

            # Add batches with stride until we reach or pass the final position
            while idx < final_idx:
                indices.append((video_idx, idx))
                idx += stride

            # Always add the final batch (avoid duplicate if already added)
            if len(indices) == 0 or indices[-1] != (video_idx, final_idx):
                indices.append((video_idx, final_idx))

        return indices

    def __iter__(self):
        # Validation sampler does not shuffle
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

class OnePerBatchDataset(torchDataset):
    def __init__(
        self,
        file_path="train_seq.npy",
        img_size=(416, 416),
        preproc=None,
        lframe = 32,
        gframe = 0,
        val = False,
        mode='random',
        dataset_pth = '',
        tnum = 1000,
        formal = False,
        traj_linking = False,
        local_stride = 1,
        class_id_map = None
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
            class_id_map (dict, optional): Mapping from XML class names to class indices.
                If None, uses default ImageNet VID mapping.
        """
        super().__init__()
        self.tnum = tnum
        self.traj_linking = traj_linking
        self.input_dim = img_size
        self.file_path = file_path
        self.arr = np.load(file_path, allow_pickle=True)
        self.batch_size = lframe
        self.mode = mode  # random, continous, uniform
        self.img_size = img_size
        self.preproc = preproc
        self.val = val
        self.formal = formal
        self.local_stride = local_stride
        self.dataset_pth = dataset_pth
        # Use custom class_id_map if provided, otherwise use default ImageNet VID mapping
        self.class_id_map = class_id_map if class_id_map is not None else name_num

    def __getitem__(self, idx):
        # idx = (which video in array, which frame of video)
        video_idx, frame_idx = idx

        # In validation mode: use stride=1 (no random stride)
        # In training mode: randomly select stride up to local_stride
        if self.val:
            stride = 1
        else:
            max_stride = min(self.local_stride, (len(self.arr[video_idx]) - frame_idx) // self.batch_size)
            stride = random.choice(range(1, max_stride + 1))

        paths = [self.arr[video_idx][i] for i in range(frame_idx, frame_idx+self.batch_size*stride, stride)]
        imgs = []
        targets = []
        imgs_info = []

        for path in paths:
            img, target, img_info, _ = self.pull_item(path)

            imgs.append(img)
            targets.append(target)
            imgs_info.append(img_info)

        if self.preproc is not None:
            imgs, targets = self.preproc(np.ascontiguousarray(imgs),
                                        targets,
                                        self.input_dim)

        return imgs, targets, imgs_info, paths, stride
    

    def get_annotation(self,path,test_size):
        path = path.replace("Data","Annotations").replace("JPEG","xml").replace("jpg", "xml").replace("PNG", "xml").replace("png", "xml")
        if os.path.isdir(path):
            files = get_xml_list(path)
        else:
            files = [path]
        files.sort()
        anno_res = []
        for xmls in files:
            photoname = xmls.replace("Annotations","Data").replace("xml","JPEG")
            file = minidom.parse(xmls)
            root = file.documentElement
            objs = root.getElementsByTagName("object")
            width = int(root.getElementsByTagName('width')[0].firstChild.data)
            height = int(root.getElementsByTagName('height')[0].firstChild.data)
            tempnode = []
            for obj in objs:
                nameNode = obj.getElementsByTagName("name")[0].firstChild.data
                xmax = int(obj.getElementsByTagName("xmax")[0].firstChild.data)
                xmin = int(obj.getElementsByTagName("xmin")[0].firstChild.data)
                ymax = int(obj.getElementsByTagName("ymax")[0].firstChild.data)
                ymin = int(obj.getElementsByTagName("ymin")[0].firstChild.data)
                x1 = np.max((0,xmin))
                y1 = np.max((0,ymin))
                x2 = np.min((width,xmax))
                y2 = np.min((height,ymax))
                if x2 >= x1 and y2 >= y1:
                    #tempnode.append((self.class_id_map[nameNode],x1,y1,x2,y2,))
                    tempnode.append(( x1, y1, x2, y2,self.class_id_map[nameNode],))
            num_objs = len(tempnode)
            res = np.zeros((num_objs, 5))
            r = min(test_size[0] / height, test_size[1] / width)
            for ix, obj in enumerate(tempnode):
                res[ix, 0:5] = obj[0:5]
            res[:, :-1] *= r
            anno_res.append(res)
        return anno_res
    
    def pull_item(self,path):
        """
                One image / label pair for the given index is picked up and pre-processed.

                Args:
                    index (int): data index

                Returns:
                    img (numpy.ndarray): pre-processed image
                    padded_labels (torch.Tensor): pre-processed label data.
                        The shape is :math:`[max_labels, 5]`.
                        each label consists of [class, xc, yc, w, h]:
                            class (float): class index.
                            xc, yc (float) : center of bbox whose values range from 0 to 1.
                            w, h (float) : size of bbox whose values range from 0 to 1.
                    info_img : tuple of h, w.
                        h, w (int): original shape of the image
                    img_id (int): same as the input index. Used for evaluation.
                """
        path = os.path.join(self.dataset_pth,path)
        annos = self.get_annotation(path, self.img_size)[0]

        img = cv2.imread(path)
        height, width = img.shape[:2]
        img_info = (height, width)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return img, annos, img_info, path

def opb_collate_fn(batch):
    # batch from OnePerBatchDataset
    # batch is a list with ONE element (since sampler returns one index per batch)
    # that element is (imgs_list, targets_list, imgs_info_list, paths_list)
    # where each list has length batch_size (number of frames in sequence)

    # Unpack the single batch item containing lists of frames
    imgs_list, targets_list, imgs_info_list, paths_list, stride = batch[0]

    # Process each frame to match the format expected by the model
    tar = []
    imgs = []
    ims_info = []
    tar_ori = []
    path = []

    for img, target, img_info, img_path in zip(imgs_list, targets_list, imgs_info_list, paths_list):
        # Create padded target tensor (max 120 objects per frame)
        tar_tensor = torch.zeros([120, 5])

        # Convert image to tensor
        # Images from pull_item are numpy arrays in HWC format
        # Need to convert to CHW tensor format for PyTorch
        if isinstance(img, np.ndarray):
            if len(img.shape) == 3 and img.shape[2] in [3, 4]:
                # HWC -> CHW
                img = np.transpose(img, (2, 0, 1))
            img = torch.from_numpy(img)

        imgs.append(img)

        # Store original targets and create padded version
        tar_ori.append(torch.tensor(target))
        if target.shape[0] > 0:
            tar_tensor[:target.shape[0]] = torch.tensor(target)
        tar.append(tar_tensor)

        # Keep metadata
        ims_info.append(img_info)
        path.append(img_path)

    # Stack into batch tensors: [batch_size, C, H, W] and [batch_size, 120, 5]
    return torch.stack(imgs), torch.stack(tar), ims_info, tar_ori, path, stride

def collate_fn(batch):
    tar = []
    imgs = []
    ims_info = []
    tar_ori = []
    path = []
    path_sequence = []
    for sample in batch:
        tar_tensor = torch.zeros([120,5])
        imgs.append(torch.tensor(sample[0]))
        tar_ori.append(torch.tensor(sample[1]))
        tar_tensor[:sample[1].shape[0]] = torch.tensor(sample[1])
        tar.append(tar_tensor)
        ims_info.append(sample[2])
        path.append(sample[3])
        #path_sequence.append(int(sample[3][sample[3].rfind('/')+1:sample[3].rfind('.')]))
    # path_sequence= torch.tensor(path_sequence)
    # time_embedding = get_timing_signal_1d(path_sequence,256)
    return torch.stack(imgs),torch.stack(tar),ims_info,tar_ori,path,None

def get_vid_loader(batch_size,data_num_workers,dataset):
    sampler = VIDBatchSampler(TrainSampler(dataset), batch_size, drop_last=False)
    dataloader_kwargs = {
        "num_workers": data_num_workers,
        "pin_memory": True,
        "batch_sampler": sampler,
        'collate_fn':collate_fn
    }
    vid_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    return vid_loader

def vid_val_loader(batch_size,data_num_workers,dataset,):
    sampler = VIDBatchSampler_Test(TestSampler(dataset),batch_size,drop_last=False)
    dataloader_kwargs = {
        "num_workers": data_num_workers,
        "pin_memory": True,
        "batch_sampler": sampler,
        'collate_fn': collate_fn
    }
    loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    return loader

def collate_fn_trans(batch):
    tar = []
    imgs = []
    ims_info = []
    tar_ori = []
    path = []
    path_sequence = []
    for sample in batch:
        tar_tensor = torch.zeros([100,5])
        imgs.append(torch.tensor(sample[0]))
        tar_ori.append(torch.tensor(copy.deepcopy(sample[1])))
        sample[1][:,1:]=xyxy2cxcywh(sample[1][:,1:])
        tar_tensor[:sample[1].shape[0]] = torch.tensor(sample[1])
        tar.append(tar_tensor)
        ims_info.append(sample[2])
        path.append(sample[3])
        path_sequence.append(int(sample[3][sample[3].rfind('/')+1:sample[3].rfind('.')]))
    path_sequence= torch.tensor(path_sequence)
    time_embedding = get_timing_signal_1d(path_sequence,256)
    return torch.stack(imgs),torch.stack(tar),ims_info,tar_ori,path,time_embedding

def get_trans_loader(batch_size,data_num_workers,dataset):
    sampler = VIDBatchSampler(TrainSampler(dataset), batch_size, drop_last=False)
    dataloader_kwargs = {
        "num_workers": data_num_workers,
        "pin_memory": True,
        "batch_sampler": sampler,
        'collate_fn':collate_fn
    }
    vid_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    return vid_loader

class DataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        self.max_iter = len(loader)
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = DataPrefetcher._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target,_,_,self.path,self.stride = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.stride = None
            self.path = None
            return

        # Disable CUDA streams to prevent NCCL deadlock in multi-GPU training
        # with torch.cuda.stream(self.stream):
        self.input_cuda()
        self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        # Disable stream synchronization to prevent NCCL deadlock
        # torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        stride = self.stride
        paths = self.path
        if input is not None:
            self.record_stream(input)
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target, paths, stride

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())

def get_timing_signal_1d(index_squence,channels,min_timescale=1.0, max_timescale=1.0e4,):
    num_timescales = channels // 2

    log_time_incre = torch.tensor(math.log(max_timescale/min_timescale)/(num_timescales-1))
    inv_timescale = min_timescale*torch.exp(torch.arange(0,num_timescales)*-log_time_incre)

    scaled_time = torch.unsqueeze(index_squence,1)*torch.unsqueeze(inv_timescale,0) #(index_len,1)*(1,channel_num)
    sig = torch.cat([torch.sin(scaled_time),torch.cos(scaled_time)],dim=1)
    return sig
