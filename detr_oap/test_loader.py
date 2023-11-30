# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# python test.py --data_path ../data_manual_annotations/images/val --resume ./checkpoint.pth --device cpu --backbone resnet101 --thresh_dist 20 --thresh 0.1 &


"""
Train and eval functions used in main.py
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment

import util.misc as utils
from datasets.pole import make_Pole_transforms
from models import build_model
from datasets.pole import PoleDetection

from tqdm import tqdm


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_backbone", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=6, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--lr_drop", default=200, type=int)
    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )

    # Model parameters
    parser.add_argument(
        "--frozen_weights",
        type=str,
        default=None,
        help="Path to the pretrained model. If set, only the mask head will be trained",
    )
    # * Backbone
    parser.add_argument(
        "--backbone",
        default="resnet50",
        type=str,
        help="Name of the convolutional backbone to use",
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )

    # * Transformer
    parser.add_argument(
        "--enc_layers", default=6, type=int, help="Number of encoding layers in the transformer"
    )
    parser.add_argument(
        "--dec_layers", default=6, type=int, help="Number of decoding layers in the transformer"
    )
    parser.add_argument(
        "--dim_feedforward",
        default=2048,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim",
        default=256,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="Dropout applied in the transformer"
    )
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions",
    )
    parser.add_argument("--num_queries", default=20, type=int, help="Number of query slots")
    parser.add_argument("--pre_norm", action="store_true")

    # * Segmentation
    parser.add_argument(
        "--masks", action="store_true", help="Train segmentation head if the flag is provided"
    )

    # # Loss
    parser.add_argument(
        "--no_aux_loss",
        dest="aux_loss",
        action="store_false",
        help="Disables auxiliary decoding losses (loss at each layer)",
    )
    # * Matcher
    parser.add_argument(
        "--set_cost_class", default=1, type=float, help="Class coefficient in the matching cost"
    )
    parser.add_argument(
        "--set_cost_dist", default=5, type=float, help="L1 box coefficient in the matching cost"
    )
    parser.add_argument(
        "--set_cost_giou", default=2, type=float, help="giou box coefficient in the matching cost"
    )
    # * Loss coefficients
    parser.add_argument("--mask_loss_coef", default=1, type=float)
    parser.add_argument("--dice_loss_coef", default=1, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument(
        "--eos_coef",
        default=0.1,
        type=float,
        help="Relative classification weight of the no-object class",
    )

    # dataset parameters
    parser.add_argument("--dataset_file", default="pole")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--data_panoptic_path", type=str)
    parser.add_argument("--remove_difficult", action="store_true")

    parser.add_argument(
        "--output_dir", default="", help="path where to save the results, empty for no saving"
    )
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument("--thresh", default=1e-10, type=float)
    parser.add_argument("--thresh_dist", default=20, type=float)
    parser.add_argument("--matching_method", default="hungarian_matching", type=str)
    parser.add_argument("--show", default=False, type=bool)

    return parser


def rescale_prediction(output, size):
    """Rescales the output keypoint coordinates to the target image size."""
    img_w, img_h = size
    res = output * torch.tensor([img_w, img_h], dtype=torch.float32)
    return res


def get_images(in_path):
    """Get all the images in the given path"""
    img_files = []
    for dirpath, dirnames, filenames in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == ".jpg" or ext == ".jpeg" or ext == ".gif" or ext == ".png" or ext == ".pgm":
                img_files.append(os.path.join(dirpath, file))

    return img_files


def get_err_sum(matching):
    """Compute the sum of the error in x"""
    error_sum_x = 0
    error_sum_l1 = 0
    error_sum_l2 = 0
    for match in matching:
        error_sum_x += abs(match[0][0] - match[1][0])
        error_sum_l1 += np.linalg.norm(match[0] - match[1], ord=1)
        error_sum_l2 += np.linalg.norm(match[0] - match[1], ord=2)

    return error_sum_x, error_sum_l1, error_sum_l2


def cost_point_to_point(pred, gt):
    """Compute the cost between two points"""
    return np.linalg.norm(pred - gt, ord=2)


def hungarian_matching(pred, gt, thresh):
    """Hungarian matching algorithm"""
    # compute the cost matrix
    cost_matrix = np.zeros((len(pred), len(gt)))

    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            if cost_point_to_point(p, g) < thresh:
                cost_matrix[i, j] = -1 / cost_point_to_point(p, g)
            else:
                cost_matrix[i, j] = 0

    # apply the hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    row_ind = list(row_ind)
    col_ind = list(col_ind)

    # when no matching add (row_ind, -1)
    for i, _ in enumerate(pred):
        if i not in row_ind:
            row_ind.append(i)
            col_ind.append(-1)

    return row_ind, col_ind


def nearest_neighbor_matching(pred, gt, thresh):
    """Nearest neighbor matching algorithm"""
    row_ind, col_ind = [], []
    gt_copy = np.copy(gt)

    for p in pred:
        for g in gt_copy:
            dist = cost_point_to_point(p, g)
            if dist <= thresh:
                row_ind.append(np.where(pred == p)[0][0])
                col_ind.append(np.where(gt == g)[0][0])
                gt_copy = np.delete(gt_copy, np.where(gt_copy == g)[0], axis=0)
                break

        row_ind.append(np.where(pred == p)[0][0])
        col_ind.append(-1)

    return row_ind, col_ind


# function that return lists of true positives, false positives and false negatives
# hungarian_matching or nearest_neighbor_matching
def get_tp_fp_fn(pred, probas, gt, dist_thresh, matching_func):
    """Get the list of true positives, false positives and false negatives"""
    l_tp, l_fp, l_fn = [], [], []
    matching = []
    pred = pred.numpy()
    if matching_func in globals():
        matching_func = globals()[matching_func]
    else:
        print(f"Function {matching_func} is not define.")

    idx = np.argsort(probas.flatten())[::-1]
    pred = pred[idx]

    row_ind, col_ind = matching_func(pred, gt, dist_thresh)

    for row_i, col_i in zip(row_ind, col_ind):
        if col_i == -1:
            l_fp.append(pred[row_i])
        elif cost_point_to_point(pred[row_i], gt[col_i]) <= dist_thresh:
            l_tp.append(pred[row_i])
            matching.append((pred[row_i], gt[col_i]))
        else:
            l_fp.append(pred[row_i])
            l_fn.append(gt[col_i])

    for g_i in range(len(gt)):
        if g_i not in col_ind:
            l_fn.append(gt[g_i])

    print(f"TP: {len(l_tp)}, FP: {len(l_fp)}, FN: {len(l_fn)}")
    return l_tp, l_fp, l_fn, matching


# get recall and precision
def get_recall_precision(tp, fp, fn):
    """Compute the recall and precision"""
    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    return recall, precision


@torch.no_grad()
def infer(dataloader, model, _, device, output_path):
    # load grount truth json from data_manual_annotations/val.json
    gt_folder = "../data_manual_annotations/final_dataset"

    model.eval()

    # intialisations values
    duration = 0
    total_tp, total_fp, total_fn = 0, 0, 0
    n_pairwise_matches = 0
    error_sum_x, error_sum_l1, error_sum_l2 = 0, 0, 0
    # array size 5xn storing the image_basename, x_pred, y_pred, x_gt, y_gt
    results = np.empty((0, 5), int)

    for samples, targets in tqdm(dataloader):
        # /!\ targets is still in shape of bbox and we only need to keep
        # the first two coordinates (x, y)$
        w, h = samples.size
        transform = make_Pole_transforms("val")
        dummy_target = {
            "size": torch.as_tensor([int(h), int(w)]),
            "orig_size": torch.as_tensor([int(h), int(w)]),
        }
        image, _ = transform(samples, dummy_target)
        image = image.unsqueeze(0).to(device)

        outputs = model(image)

        outputs["pred_logits"] = outputs["pred_logits"].cpu()
        outputs["pred_boxes"] = outputs["pred_boxes"].cpu()

        probas = outputs["pred_logits"].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values >= 0.5

        # Scale keypoints to the original image size
        points_scaled = rescale_prediction(outputs["pred_boxes"][0, keep], samples.size)
        probas = probas[keep].cpu().data.numpy()

        gt_data = targets["boxes"].cpu().data.numpy()[:, :2]
        l_tp, l_fp, l_fn, matching = get_tp_fp_fn(
            points_scaled, probas, gt_data, 10, "hungarian_matching"
        )

        # add the matching results to the results array
        # for match in matching:
        #     results = np.append(
        #         results,
        #         [[file_basename, match[0][0], match[0][1], match[1][0], match[1][1]]],
        #         axis=0,
        #     )

        n_pairwise_matches += len(matching)
        err_x, err_l1, err_l2 = get_err_sum(matching)
        error_sum_x += err_x
        error_sum_l1 += err_l1
        error_sum_l2 += err_l2

        total_tp += len(l_tp)
        total_fp += len(l_fp)
        total_fn += len(l_fn)

        if args.show:
            print(gt_data)
            # if len(l_tp) + len(l_fp) + len(l_fn) == 0:
            #     continue

            img = np.array(samples)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # for tp in l_tp:
            #     cv2.circle(img, (int(tp[0]), int(tp[1])), 2, (0, 0, 255), 2)  # red

            # for fp in l_fp:
            #     cv2.circle(img, (int(fp[0]), int(fp[1])), 2, (255, 0, 0), 2)  # blue

            # for fn in l_fn:
            #     cv2.circle(img, (int(fn[0]), int(fn[1])), 2, (0, 255, 0), 2)  # green

            # draw lines between the matching points
            for match in matching:
                cv2.line(
                    img,
                    (int(match[0][0]), int(match[0][1])),
                    (int(match[1][0]), int(match[1][1])),
                    (0, 0, 255),
                    1,
                )

            for gt in gt_data:
                cv2.circle(img, (int(gt[0]), int(gt[1])), 2, (0, 255, 0), 2)

            for pred in points_scaled:
                cv2.circle(img, (int(pred[0]), int(pred[1])), 2, (0, 0, 255), 2)

            # img_save_path = os.path.join(output_path, filename)
            # cv2.imwrite(img_save_path, img)
            cv2.imshow("img", img)
            cv2.waitKey()

        # print(f"End processing {filename}")

    # compute precision and recall
    recall, precision = get_recall_precision(total_tp, total_fp, total_fn)
    MAE_x = error_sum_x / n_pairwise_matches
    MAE_l1 = error_sum_l1 / n_pairwise_matches
    MAE_l2 = error_sum_l2 / n_pairwise_matches

    # print("Avg. Time: {:.3f}s".format(avg_duration))
    print("Recall: {:.3f}, Precision: {:.3f}".format(recall, precision))
    print("MAE_x: {:.3f}, MAE_l1: {:.3f}, MAE_l2: {:.3f}".format(MAE_x, MAE_l1, MAE_l2))

    # save the results in a csv file with a header (basename, x_pred, y_pred, x_gt, y_gt)
    csv_file = os.path.join(output_path, "matching_results.csv")
    np.savetxt(
        csv_file, results, delimiter=",", fmt="%s", header="basename,x_pred,y_pred,x_gt,y_gt"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DETR training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    model, _, postprocessors = build_model(args)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    model.to(device)
    image_paths = get_images(args.data_path)
    val_dataset = PoleDetection(
        "../data_manual_annotations/",
        "../datasets/default_bdd100k_val/val.json",
        transforms=None,
        return_masks=args.masks,
    )


    infer(val_dataset, model, postprocessors, device, args.output_dir)
