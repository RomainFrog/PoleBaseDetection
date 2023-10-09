# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import cv2
import sys
import argparse
from pathlib import Path
from typing import Iterable
from PIL import Image
import numpy as np

import torch

import util.misc as utils

from models import build_model
from datasets.pole import make_Pole_transforms
from datasets.pole import PoleDetection

import matplotlib.pyplot as plt
import time
import torchvision.transforms as T
from tqdm import tqdm



def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h,
                          img_w, img_h
                          ], dtype=torch.float32)
    return b

def get_images(in_path):
    img_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))

    return img_files

def iou(boxA, boxB):
    """ Compute bbox IoU. """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


# function that return number of true positives, false positives and false negatives
def get_tp_fp_fn(pred, probas, gt, thresh):
    l_tp, l_fp, l_fn = [], [], []
    matching = []

    pred = pred.numpy()
    # print(pred)
    # print(probas)
    idx = np.argsort(probas.flatten())[::-1]
    pred = pred[idx]
    # print(pred)
    # print(gt)

    pred_copy = np.copy(pred)
    gt_copy = np.copy(gt)
    # print(type(gt_copy))
    for p in pred_copy:
        for g in gt_copy:
            bbox_iou = iou(p, g)
            # Pass if IoU is less than the threshold (tipically 0.5)
            if bbox_iou >= 0.35:
                gt_copy = np.delete(gt_copy, np.where(gt_copy == g)[0], axis=0)
                pred_copy = np.delete(pred_copy, np.where(pred_copy == p)[0], axis=0)
                l_tp.append(p)
                matching.append((p, g))
                break
        else:
            l_fp.append(p)

    for g in gt_copy:
        l_fn.append(g)

    # print("tp: {}, fp: {}, fn: {}".format(len(l_tp), len(l_fp), len(l_fn)))

    return l_tp, l_fp, l_fn, matching


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=6, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=20, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='pole')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--data_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save the results, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument('--thresh', default=0.5, type=float)

    parser.add_argument("--show", default=False, type=bool)


    return parser


# get recall and precision
def get_recall_precision(tp, fp, fn):
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)

    return recall, precision


def get_MAE(matching):
    """ Compute MAE_x, MAE_l1, MAE_l2 sums. """
    err_x_sum, err_l1_sum, err_l2_sum = 0,0,0
    for match in matching:
        pred, gt = match
        pred_x = (pred[0] + pred[2]) / 2
        pred_y = (pred[1] + pred[3]) / 2
        gt_x = (gt[0] + gt[2]) / 2
        gt_y = (gt[1] + gt[3]) / 2
        err_x_sum += abs(pred_x - gt_x)
        pred_center = np.array([pred_x, pred_y])
        gt_center = np.array([gt_x, gt_y])
        # Compute L1 norm
        err_l1_sum += np.linalg.norm(pred_center - gt_center, ord=1)
        # Compute L2 norm
        err_l2_sum += np.linalg.norm(pred_center - gt_center, ord=2)

    return err_x_sum, err_l1_sum, err_l2_sum

        


@torch.no_grad()
def infer(images_path, model, postprocessors, device, dataset):
    # load grount truth json from data_manual_annotations/val.json
    model.eval()

    ##### Auxiliary variables #####
    duration, total_tp, total_fp, total_fn = 0,0,0,0
    n_pairwise_matches, error_sum_x, error_sum_l1, error_sum_l2 = 0,0,0,0
    ###############################
    
    for orig_image, target in tqdm(dataset):
        # cast orig tensor to PIL image
        w, h = orig_image.size
        transform = make_Pole_transforms("val")
        dummy_target = {
            "size": torch.as_tensor([int(h), int(w)]),
            "orig_size": torch.as_tensor([int(h), int(w)]),
        }
        image, _ = transform(orig_image, dummy_target)
        image = image.unsqueeze(0)
        image = image.to(device)


        start_t = time.perf_counter()
        outputs = model(image)
        end_t = time.perf_counter()

        outputs["pred_logits"] = outputs["pred_logits"].cpu()
        outputs["pred_boxes"] = outputs["pred_boxes"].cpu()

        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        # keep = probas.max(-1).values > 0.85
        keep = probas.max(-1).values > 0.5

        gt_data = target['boxes']
        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], (w,h))
        probas = probas[keep].cpu().data.numpy()

        # print("Start matching")
        l_tp, l_fp, l_fn, matching = get_tp_fp_fn(bboxes_scaled, probas, gt_data, args.thresh)
        # print("End matching")

        n_pairwise_matches += len(matching)
        #TODO: get error sum
        # err_x, err_l1, err_l2 = get_err_sum(matching)
        # error_sum_x += err_x
        # error_sum_l1 += err_l1
        # error_sum_l2 += err_l2

        total_tp += len(l_tp)
        total_fp += len(l_fp)
        total_fn += len(l_fn)


        if args.show:
            if len(bboxes_scaled) == 0:
                # print("No detection")
                continue
            ###### Plot prediction ######
            img = np.array(orig_image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for idx, box in enumerate(bboxes_scaled):
                bbox = box.cpu().data.numpy()
                bbox = bbox.astype(np.int32)
                bbox = np.array([
                    [bbox[0], bbox[1]],
                    [bbox[2], bbox[1]],
                    [bbox[2], bbox[3]],
                    [bbox[0], bbox[3]],
                    ])
                bbox = bbox.reshape((4, 2))
                cv2.polylines(img, [bbox], True, (0, 0, 255), 2)
                # Display a RED dot in the center of the bounding box
                # center = (int((bbox[0][0] + bbox[2][0]) / 2), int((bbox[0][1] + bbox[2][1]) / 2))
                # cv2.circle(img, center, 2, (0, 0, 255), 2)

            ##### Plot ground truth #####
            for idx, box in enumerate(gt_data):
                bbox = box.cpu().data.numpy()
                bbox = bbox.astype(np.int32)
                bbox = np.array([
                    [bbox[0], bbox[1]],
                    [bbox[2], bbox[1]],
                    [bbox[2], bbox[3]],
                    [bbox[0], bbox[3]],
                    ])
                bbox = bbox.reshape((4, 2))
                cv2.polylines(img, [bbox], True, (0, 255, 0), 2)

            cv2.imshow("img", img)
            cv2.waitKey()
        infer_time = end_t - start_t
        duration += infer_time
        # print("Processing... ({:.3f}s)".format(infer_time))

    # compute precision and recall
    # TODO: get MAE
    recall, precision = get_recall_precision(total_tp, total_fp, total_fn)
    error_sum_x, error_sum_l1, error_sum_l2 = get_MAE(matching)
    MAE_x = error_sum_x / n_pairwise_matches
    MAE_l1 = error_sum_l1 / n_pairwise_matches
    MAE_l2 = error_sum_l2 / n_pairwise_matches

    print("Recall: {:.3f}, Precision: {:.3f}".format(recall, precision))
    print("MAE_x: {:.3f}, MAE_l1: {:.3f}, MAE_l2: {:.3f}".format(MAE_x, MAE_l1, MAE_l2))
    avg_duration = duration / len(images_path)
    print("Avg. Time: {:.3f}s".format(avg_duration))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    device = torch.device(args.device)

    model, _, postprocessors = build_model(args)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    model.to(device)
    image_paths = get_images(args.data_path)

    # Create PoleDataset
    dataset_folder = args.data_path
    dataset_test = PoleDetection(dataset_folder + "/images", dataset_folder + "/val.json",transforms=None, return_masks=args.masks)

    infer(image_paths, model, postprocessors, device, dataset_test)
