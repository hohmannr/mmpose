import os.path as osp
import json

import torch
import pandas as pd
import numpy as np
import cv2
import mmcv
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from tqdm import tqdm

from mmpose.apis import init_pose_model
from mmpose.models import build_posenet
from mmpose.datasets import build_dataset, build_dataloader

# config constants
RESNET101_COCO_256x192_CKP = 'https://download.openmmlab.com/mmpose/top_down/resnet/res101_coco_256x192-6e6babf0_20200708.pth'
RESNET101_COCO_256x192_CFG = 'configs/top_down/resnet/coco/res101_coco_256x192.py'
MODEL_IN_SIZE = 256, 192


def load_mmpose_config(cfg_file):
    """
    Loads a mmcv.Config from given file path.

    Arguments:
        cfg_file    path to model's and dataset's mmpose config file

    Returns:
        cfg         mmcv.Config
    """
    cfg = mmcv.Config.fromfile(cfg_file)

    # prepare for inference
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    return cfg


def load_model(cfg, ckp_link):
    """
    Loads a model from given mmcv.Config file and the related weights from the checkpoint link.

    Arguments:
        cfg         mmcv.Config for the model
        ckp_link    link to download the model's weights from

    Returns:
        model       nn.Module
    """
    model = build_posenet(cfg.model)
    load_checkpoint(model, ckp_link, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0]) # device id, just set first cuda GPU

    return model


def inference_single_frame(model, data, return_heatmap=False, beautify_results=True):
    """
    Does inference for model of a given a single data object.

    Arguments:
        model   MMDataParallel/nn.Module in eval mode
        data    dict(
                    img         torch.Tensor
                    img_metas   mmcv.DataContainer
                )
        return_heatmap      bool    decides if keypoint heatmaps shall be returned
        beautify_results    bool    decides if the vanilla return from mmpose should be used or if results will be beatified (human readable)

    Desc:
        'img_metas' DataContainer consists of
            [[
                dict(
                    image_file  str
                    center      np.ndarray of shape (2,)
                    scale       np.ndarray of shape (2,)
                    rotation    float
                    bbox_score  float
                    flip_pairs  list of list of ints of shape (16, 2)
                )
            ]]

    Returns:
        results     inference results, can be beautified as a dict or mmpose vanilla result format
    """
    with torch.no_grad():
        result = model(return_heatmap=return_heatmap, return_loss=False, **data)

    if not beautify_results:
        return result

    # beautify results
    results = dict(
                keypoints = result[0],
                unknown = result[1],
                image_file = "".join(result[2])
            )

    if return_heatmap:
        results['heatmap'] = result[3]

    return results


def inference(model, dataloader, return_heatmap=False, beautify_results=True):
    """
    Does inference step for model from given dataloader.
    """
    model.eval()
    dataset = dataloader.dataset

    results = []
    # TODO: remove count
    c = 0
    for data in tqdm(dataloader):
        result = inference_single_frame(model, data, beautify_results=beautify_results)
        results.append(result)

        img_meta = _get_image_meta(data)

        # TODO: remove show image part
        img = cv2.imread(result['image_file'])
        # TODO: remove hardcoded image size
        model_in_size = MODEL_IN_SIZE
        bbox = bbox_get_xywh(img_meta['center'], img_meta['scale'], model_in_size)
        img = bbox_draw(img, bbox)
        img = keypoints_draw(result['keypoints'], img)

        # TODO: remove break
        cv2.imshow('img', img)
        cv2.waitKey(0)
        
        if c == 20:
            break
        else:
            c += 1

    return results


def _get_image_meta(data):
    """
    Gets image meta data (filepath, bounding box center, scale etc.) from data object.
    """
    return data['img_metas'].data[0][0]


def bbox_get_xywh(center, scale, model_in_size):
    """
    Gets bounding box top left coordinates aswell as its width and height from the bounding box center and its scale related to the model input size.

    Arguments:
        center          list of center cooridnates as [cx, cy]
        scale           list of scaling done on each axis as [sx, sy]
        model_in_size   list of model input size as [h, w]

    Returns:
        bbox            tuple of (x, y, w, h)
    """
    in_h, in_w = model_in_size
    sx, sy = scale
    cx, cy = center

    # calculate bbox
    bbox_w, bbox_h = int(in_w * sx), int(in_h * sy)
    bbox_x, bbox_y = int(cx - bbox_w/2), int(cy - bbox_h/2)

    return bbox_x, bbox_y, bbox_w, bbox_h


def bbox_draw(img, bbox, color=(0, 0, 255), width=2):
    """
    Draws a bounding box from its given center and its scale related to the models input size onto given image.

    Arguments:
        img     cv2.img to be drawn upon
        bbox    tuple of of bounding box as top left coords and wdith and height (x, y, w, h)
        color   in BGR-Format, default is red
        width   line width of bounding box rectangle

    Returns:
        img     cv2.img that has bounding box drawn on it
    """
    bbox_x, bbox_y, bbox_w, bbox_h = bbox

    return cv2.rectangle(img, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), color, width)


def bbox_crop(img, bbox):
    """
    Crops the bounding box from given image. Its given the bounding box center and its scale related to the models input size.

    Arguments:
        img     cv2.img to be drawn upon
        bbox    tuple of of bounding box as top left coords and wdith and height (x, y, w, h)

    Returns:
        img     cv2.img which is the cropped bounding box
    """
    bbox_x, bbox_y, bbox_w, bbox_h = bbox

    from_x, from_y = bbox_x, bbox_y
    to_x, to_y = bbox_x + bbox_w, bbox_y + bbox_h

    return img[from_y:to_y, from_x:to_x, :]


def keypoints_draw(keypoints, img, threshold=0.3, color=(0, 255, 0), width=2):
    """
    Draws keypoints on image.

    Arguments:
        keypoints       np.ndarray in absolute coordinates of shape (1, 17, 3)
        img             cv2.img keypoints are drawn upon
        threshold       keypoint score threshold to surpass before keypoint is drawn
        color           in BGR-Format, default is green
        width           width of circle outline

    Returns:
        img             cv2.img that has keypoints drawn on it
    """
    for kp in keypoints[0]:
        x, y, score = kp
        if score >= threshold:
            img = cv2.circle(img, (x, y), 2, color, width)

    return img


def keypoints_get_annotations(idx, annotation_file, bbox_file):
    """
    Takes an (dataframe) index and returns image information (filepath, downloads, etc.) and keypoint annotation info and bounding box info.
    """
    with open(annotation_file) as f:
        data = json.load(f)

    img_record = data['images']
    annot_record = data['annotations']

    img_df = pd.DataFrame.from_records(img_record)
    annot_df = pd.DataFrame.from_records(annot_record)
    bbox_df = pd.read_json(bbox_file, orient='records')

    assert idx >= 0 and idx < img_df.shape[0], "error: image idx not found."

    # get correct a coco image object and its keypoint annotation and prelocated bboxes
    img_obj = img_df.iloc[idx, :]
    annot_obj = annot_df.loc[annot_df['image_id'] == img_obj['id']]
    bbox_obj = bbox_df.loc[bbox_df['image_id'] == img_obj['id']]

    return img_obj, annot_obj, bbox_obj


def gt_bboxes_draw(img, bboxes):
    """
    Draws given pd.DataFrame of boundingboxes on given cv2.img.
    """
    for bb in bboxes['bbox']:
        x, y, w, h = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
        color = (255, 255, 0)
        line_thickness = 2
        img = cv2.rectangle(img, (x, y), (x + w, y + h), color, line_thickness)

    return img


def gt_keypoints_draw(img, keypoints):
    """
    Draws given pd.DataFrame of keypoints on given cv2.img.
    """
    for kps in keypoints['keypoints']:
        for i in range(0, len(kps), 3):
            curr_kp = []
            for j in range(2):
                curr_kp.append(kps[i + j])
            img = cv2.circle(img, (int(curr_kp[0]), int(curr_kp[1])), 2, (0, 0, 255), 2)

    return img
                

# load model
cfg = load_mmpose_config(RESNET101_COCO_256x192_CFG)
model = load_model(cfg, RESNET101_COCO_256x192_CKP)

# load data
dataset = build_dataset(cfg.data.test, dict(test_mode=True))
dataloader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=1, #cfg.data.workers_per_gpu,
        dist=False,
        shuffle=True)

results = inference(model, dataloader, return_heatmap=True)
