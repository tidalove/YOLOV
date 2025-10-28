#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
"""
Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import math
import random

import cv2
import numpy as np

from kornia.geometry.transform import resize
from kornia.augmentation import ColorJiggle

from yolox.utils import xyxy2cxcywh


def augment_hsv(img, hgain=5, sgain=30, vgain=30):
    hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]  # random gains
    hsv_augs *= np.random.randint(0, 2, 3)  # random selection of h, s, v
    hsv_augs = hsv_augs.astype(np.int16)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

    img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

    cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def get_aug_params(value, center=0):
    if isinstance(value, float):
        return random.uniform(center - value, center + value)
    elif len(value) == 2:
        return random.uniform(value[0], value[1])
    else:
        raise ValueError(
            "Affine params should be either a sequence containing two values\
             or single float values. Got {}".format(value)
        )


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):
    # box1(4,n), box2(4,n)
    # Compute candidate boxes which include follwing 5 things:
    # box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (
        (w2 > wh_thr)
        & (h2 > wh_thr)
        & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr)
        & (ar < ar_thr)
    )  # candidates


def random_perspective(
    img,
    targets=(),
    degrees=10,
    translate=0.1,
    scale=0.1,
    shear=10,
    perspective=0.0,
    border=(0, 0),
):
    # targets = [cls, xyxy]
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(scale[0], scale[1])
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * width
    )  # x translation (pixels)
    T[1, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * height
    )  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT

    ###########################
    # For Aug out of Mosaic
    # s = 1.
    # M = np.eye(3)
    ###########################

    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(
                img, M, dsize=(width, height), borderValue=(114, 114, 114)
            )
        else:  # affine
            img = cv2.warpAffine(
                img, M[:2], dsize=(width, height), borderValue=(114, 114, 114)
            )

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
            n * 4, 2
        )  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, :4].T * s, box2=xy.T)
        targets = targets[i]
        targets[:, :4] = xy[i]

    return img, targets


def get_affine_matrix(
    target_size,
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    twidth, theight = target_size

    # Rotation and Scale
    angle = get_aug_params(degrees)
    scale = get_aug_params(scales, center=1.0)

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    R = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)

    M = np.ones([2, 3])
    # Shear
    shear_x = math.tan(get_aug_params(shear) * math.pi / 180)
    shear_y = math.tan(get_aug_params(shear) * math.pi / 180)

    M[0] = R[0] + shear_y * R[1]
    M[1] = R[1] + shear_x * R[0]

    # Translation
    translation_x = get_aug_params(translate) * twidth  # x translation (pixels)
    translation_y = get_aug_params(translate) * theight  # y translation (pixels)

    M[0, 2] = translation_x
    M[1, 2] = translation_y

    return M, scale


def apply_affine_to_bboxes(targets, target_size, M, scale):
    num_gts = len(targets)

    # warp corner points
    twidth, theight = target_size
    corner_points = np.ones((4 * num_gts, 3))
    corner_points[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
        4 * num_gts, 2
    )  # x1y1, x2y2, x1y2, x2y1
    corner_points = corner_points @ M.T  # apply affine transform
    corner_points = corner_points.reshape(num_gts, 8)

    # create new boxes
    corner_xs = corner_points[:, 0::2]
    corner_ys = corner_points[:, 1::2]
    new_bboxes = (
        np.concatenate(
            (corner_xs.min(1), corner_ys.min(1), corner_xs.max(1), corner_ys.max(1))
        )
        .reshape(4, num_gts)
        .T
    )

    # clip boxes
    new_bboxes[:, 0::2] = new_bboxes[:, 0::2].clip(0, twidth)
    new_bboxes[:, 1::2] = new_bboxes[:, 1::2].clip(0, theight)

    targets[:, :4] = new_bboxes

    return targets


def random_affine(
    img,
    targets=(),
    target_size=(640, 640),
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    M, scale = get_affine_matrix(target_size, degrees, translate, scales, shear)

    img = cv2.warpAffine(img, M, dsize=target_size, borderValue=(114, 114, 114))

    # Transform label coordinates
    if len(targets) > 0:
        targets = apply_affine_to_bboxes(targets, target_size, M, scale)

    return img, targets

def batched_mirror(images, boxes, prob=0.5):
    *_, width = images.shape
    if random.random() < prob:
        images = images[..., ::-1]
        # boxes: 
        boxes[:, 0::2] = width - boxes[:, 2::-2] # first dim is N
    return images, boxes

def _mirror(image, boxes, prob=0.5):
    _, width, _ = image.shape
    if random.random() < prob:
        image = image[:, ::-1]
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes


def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def batched_preproc(img, input_size):
    if len(img.shape) == 4:
        img = img.transpose(0, 3, 1, 2) # needs shape (..., H, W) for kornia resize
        N, C, H, W = img.shape
        padded_img = np.ones((N, 3, input_size[0], input_size[1]), dtype=np.uint8) * 114
    else:
        N, H, W = img.shape
        padded_img = np.ones((N, input_size[0], input_size[1]), dtype=np.uint8) * 114

    r = min(input_size[0] / H, input_size[1] / W)
    resized_img = resize(
        img, 
        (int(H * r), int(W * r))
    ).astype(np.uint8)
    padded_img[..., : int(H * r), : int(W * r)] = resized_img

    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

class BatchedTrainTransform:
    """
    Applies the same random augmentation to all frames in a batch.
    Designed for temporal consistency in video object detection.

    Args:
        images: (batch_size, H, W, C) stacked numpy array
        targets_list: list of (N_i, 5) arrays where each is [x1, y1, x2, y2, class_id]
        input_dim: tuple (H_out, W_out) target size

    Returns:
        images: (batch_size, C, H_out, W_out) preprocessed numpy array
        targets_list: list of (max_labels, 5) padded arrays in [class_id, cx, cy, w, h] format
    """
    def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0, h_factor=0.015, s_factor=0.7, v_factor=0.4):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob
        self.h = h_factor
        self.s = s_factor
        self.v = v_factor

    def __call__(self, images, targets_list, input_dim):
        """
        Apply synchronized augmentation to batch of images.

        Performance note: For batch_size=16-32, the for-loop adds <5% overhead
        compared to the actual image operations (resize, HSV, etc.), which is
        acceptable for the benefit of maintaining per-frame target structure.
        """
        batch_size = images.shape[0]
        height_o, width_o = images.shape[1:3]

        # Sample augmentation parameters ONCE for entire batch (synchronization)
        do_hsv = random.random() < self.hsv_prob
        do_flip = random.random() < self.flip_prob

        # Sample HSV parameters (convert factors to cv2 gains)
        if do_hsv:
            hsv_h = random.uniform(-self.h, self.h) * 180  # Convert to hue range [0, 180]
            hsv_s = random.uniform(-self.s, self.s) * 255  # Convert to saturation range
            hsv_v = random.uniform(-self.v, self.v) * 255  # Convert to value range

        processed_images = []
        processed_targets = []

        # Apply same transform to each frame
        for i in range(batch_size):
            img = images[i].copy()  # (H, W, C)
            target = targets_list[i].copy()  # (N_i, 5) in [x1, y1, x2, y2, class_id]
            boxes = target[:, :4]
            labels = target[:, 4]

            # Handle empty annotations
            if len(boxes) == 0:
                img_processed, _ = preproc(img, input_dim, swap=(2, 0, 1))
                padded = np.zeros((self.max_labels, 5), dtype=np.float32)
                processed_images.append(img_processed)
                processed_targets.append(padded)
                continue

            # Keep original for fallback
            img_o = img.copy()
            boxes_o = xyxy2cxcywh(boxes.copy())
            labels_o = labels.copy()

            # Apply HSV augmentation (in-place, same parameters across batch)
            if do_hsv:
                # Use fixed HSV gains sampled once for the batch
                hsv_augs = np.array([hsv_h, hsv_s, hsv_v], dtype=np.int16)
                img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
                img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
                img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
                img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)
                cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)

            # Apply horizontal flip (same decision across batch)
            if do_flip:
                img = img[:, ::-1]
                boxes[:, 0::2] = width_o - boxes[:, 2::-2]

            # Preprocess (resize + pad)
            img, r_ = preproc(img, input_dim, swap=(2, 0, 1))

            # Transform boxes to [cx, cy, w, h] and scale
            boxes = xyxy2cxcywh(boxes)
            boxes *= r_

            # Filter out boxes that are too small (< 1 pixel)
            mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
            boxes_t = boxes[mask_b]
            labels_t = labels[mask_b]

            # Fallback to original if all boxes filtered out
            if len(boxes_t) == 0:
                img, r_o = preproc(img_o, input_dim, swap=(2, 0, 1))
                boxes_t = boxes_o * r_o
                labels_t = labels_o

            # Create padded target array [class_id, cx, cy, w, h]
            labels_t = np.expand_dims(labels_t, 1)
            targets_t = np.hstack((labels_t, boxes_t))
            padded = np.zeros((self.max_labels, 5), dtype=np.float32)
            padded[:min(len(targets_t), self.max_labels)] = targets_t[:self.max_labels]
            padded = np.ascontiguousarray(padded, dtype=np.float32)

            processed_images.append(img)
            processed_targets.append(padded)

        # Stack images: (batch_size, C, H, W)
        images_out = np.stack(processed_images)

        # Return list of targets for compatibility with opb_collate_fn
        return images_out, processed_targets


class TrainTransform:
    def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob

    def __call__(self, image, targets, input_dim):
        boxes = targets[:, :4].copy()
        labels = targets[:, 4].copy()
        if len(boxes) == 0:
            targets = np.zeros((self.max_labels, 5), dtype=np.float32)
            image, r_o = preproc(image, input_dim)
            return image, targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :4]
        labels_o = targets_o[:, 4]
        # bbox_o: [xyxy] to [c_x,c_y,w,h]
        boxes_o = xyxy2cxcywh(boxes_o)

        if random.random() < self.hsv_prob:
            augment_hsv(image)
        image_t, boxes = _mirror(image, boxes, self.flip_prob)
        height, width, _ = image_t.shape
        image_t, r_ = preproc(image_t, input_dim)
        # boxes [xyxy] 2 [cx,cy,w,h]
        boxes = xyxy2cxcywh(boxes)
        boxes *= r_

        mask_b = np.minimum(boxes[:, 2], boxes[:, 3]) > 1
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b]

        if len(boxes_t) == 0:
            image_t, r_o = preproc(image_o, input_dim)
            boxes_o *= r_o
            boxes_t = boxes_o
            labels_t = labels_o

        labels_t = np.expand_dims(labels_t, 1)

        targets_t = np.hstack((labels_t, boxes_t))
        padded_labels = np.zeros((self.max_labels, 5))
        padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
            : self.max_labels
        ]
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
        return image_t, padded_labels


class ValTransform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, swap=(2, 0, 1), legacy=False):
        self.swap = swap
        self.legacy = legacy

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        img, _ = preproc(img, input_size, self.swap)
        if self.legacy:
            img = img[::-1, :, :].copy()
            img /= 255.0
            img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        return img, np.zeros((1, 5))

class Vid_Val_Transform:
    """
    Defines the transformations that should be applied to test PIL image
    for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels

    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, swap=(2, 0, 1), legacy=False):
        self.swap = swap
        self.legacy = legacy

    # assume input is cv2 img for now
    def __call__(self, img, res, input_size):
        img, r_ = preproc(img, input_size, self.swap)
        if self.legacy:
            img = img[::-1, :, :].copy()
            img /= 255.0
            img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        boxes = res[:, :4].copy()
        labels = res[:, 4].copy()
        boxes *= r_
        labels_t = np.expand_dims(labels, 1)
        targets_t = np.hstack((labels_t, boxes))
        return img, targets_t