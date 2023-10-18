# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
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
import util.misc as utils
from datasets.pole import make_Pole_transforms
from models import build_model
from PIL import Image
from scipy.optimize import linear_sum_assignment


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = out_bbox * torch.tensor([img_w, img_h], dtype=torch.float32)
    return b


def get_images(in_path):
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
        print(match)
        error_sum_x += abs(match[0][0] - match[1][0])
        error_sum_l1 += np.linalg.norm(match[0] - match[1], ord=1)
        error_sum_l2 += np.linalg.norm(match[0] - match[1], ord=2)

    return error_sum_x, error_sum_l1, error_sum_l2


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

    parser.add_argument("--thresh", default=0.5, type=float)

    parser.add_argument("--show", default=False, type=bool)

    return parser


@torch.no_grad()
def infer(images_path, model, postprocessors, device, output_path):
    # load grount truth json from data_manual_annotations/val.json
    gt = "../data_manual_annotations/annotations_tx_reviewed_final"

    model.eval()
    duration = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    total = 0
    n_pairwise_matches = 0
    error_sum_x = 0
    error_sum_l1 = 0
    error_sum_l2 = 0
    # create an array that will be of size 5xn storing the image basename, x_pred, y_pred, x_gt, y_gt (we dont know the size of n yet)
    results = np.empty((0, 5), int)



    for img_sample in images_path:
        filename = os.path.basename(img_sample)
        # get the file name without extension
        file_basename = os.path.splitext(filename)[0]
        # get ground truth from csv file
        gt_file = os.path.join(gt, file_basename + ".csv")
        if not os.path.exists(gt_file):
            continue

        print("processing...{}".format(filename))
        # if the file doesn't have any ground truth
        if os.path.getsize(gt_file) <= 14:
            gt_data = np.array([])
        else:
            # read csv file data (skip first line)
            gt_data = np.genfromtxt(gt_file, delimiter=",", skip_header=1, usecols=[1, 2]).astype(
                np.int32
            )
            if len(gt_data.shape) == 1:
                gt_data = np.expand_dims(gt_data, axis=0)

        orig_image = Image.open(img_sample)
        w, h = orig_image.size
        transform = make_Pole_transforms("val")
        dummy_target = {
            "size": torch.as_tensor([int(h), int(w)]),
            "orig_size": torch.as_tensor([int(h), int(w)]),
        }
        image, targets = transform(orig_image, dummy_target)
        image = image.unsqueeze(0)
        image = image.to(device)

        conv_features, enc_attn_weights, dec_attn_weights = [], [], []
        hooks = [
            model.backbone[-2].register_forward_hook(
                lambda self, input, output: conv_features.append(output)
            ),
            model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            ),
            model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
                lambda self, input, output: dec_attn_weights.append(output[1])
            ),
        ]

        print("Start inference")
        start_t = time.perf_counter()
        outputs = model(image)
        end_t = time.perf_counter()
        infer_time = end_t - start_t
        duration += infer_time
        print("End inference ({:.3f}s)".format(infer_time))

        outputs["pred_logits"] = outputs["pred_logits"].cpu()
        outputs["pred_boxes"] = outputs["pred_boxes"].cpu()

        probas = outputs["pred_logits"].softmax(-1)[0, :, :-1]
        # keep = probas.max(-1).values > 0.85
        keep = probas.max(-1).values > args.thresh

        # Scale bbox to the same size as the original
        bboxes_scaled = rescale_bboxes(outputs["pred_boxes"][0, keep], orig_image.size)
        probas = probas[keep].cpu().data.numpy()

        for hook in hooks:
            hook.remove()

        conv_features = conv_features[0]
        enc_attn_weights = enc_attn_weights[0]
        dec_attn_weights = dec_attn_weights[0].cpu()

        # get the feature map shape
        h, w = conv_features["0"].tensors.shape[-2:]

        print("Start matching")
        l_tp, l_fp, l_fn, matching = get_tp_fp_fn(bboxes_scaled, probas, gt_data, 10)
        print("End matching")
        
        # add the matching results to the results array
        for match in matching:
            results = np.append(results, [[file_basename, match[0][0], match[0][1], match[1][0], match[1][1]]], axis=0)

        n_pairwise_matches += len(matching)
        err_x, err_l1, err_l2 = get_err_sum(matching)
        error_sum_x += err_x
        error_sum_l1 += err_l1
        error_sum_l2 += err_l2

        total_tp += len(l_tp)
        total_fp += len(l_fp)
        total_fn += len(l_fn)

        if args.show:
            if len(l_tp) + len(l_fp) + len(l_fn) == 0:
                continue

            img = np.array(orig_image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for tp in l_tp:
                cv2.circle(img, (int(tp[0]), int(tp[1])), 2, (0, 0, 255), 2)  # red

            for fp in l_fp:
                cv2.circle(img, (int(fp[0]), int(fp[1])), 2, (255, 0, 0), 2)  # blue

            for fn in l_fn:
                cv2.circle(img, (int(fn[0]), int(fn[1])), 2, (0, 255, 0), 2)  # green

            # img_save_path = os.path.join(output_path, filename)
            # cv2.imwrite(img_save_path, img)
            cv2.imshow("img", img)
            cv2.waitKey()

        print(f"End processing {filename}")

    # compute precision and recall
    recall, precision = get_recall_precision(total_tp, total_fp, total_fn)
    MAE_x = error_sum_x / n_pairwise_matches
    MAE_l1 = error_sum_l1 / n_pairwise_matches
    MAE_l2 = error_sum_l2 / n_pairwise_matches

    avg_duration = duration / len(images_path)
    print("Avg. Time: {:.3f}s".format(avg_duration))
    print("Recall: {:.3f}, Precision: {:.3f}".format(recall, precision))
    print("MAE_x: {:.3f}, MAE_l1: {:.3f}, MAE_l2: {:.3f}".format(MAE_x, MAE_l1, MAE_l2))

    # save the results in a csv file with a header (basename, x_pred, y_pred, x_gt, y_gt)
    csv_file = os.path.join(output_path, "matching_results.csv")
    np.savetxt(csv_file, results, delimiter=",", fmt="%s", header="basename,x_pred,y_pred,x_gt,y_gt")


def cost_point_to_point(pred, gt):
    return np.linalg.norm(pred - gt)


# function that return number of true positives, false positives and false negatives
def get_tp_fp_fn(pred, probas, gt, thresh):
    l_tp, l_fp, l_fn = [], [], []
    matching = []

    pred = pred.numpy()
    print(pred)
    print(probas)
    idx = np.argsort(probas.flatten())[::-1]
    pred = pred[idx]

    cost_matrix = np.zeros((len(pred), len(gt)))

    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            if cost_point_to_point(p, g) < thresh:
                cost_matrix[i, j] = -1
            else:
                cost_matrix[i, j] = 0

    # apply the hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    for row_i, col_i in zip(row_ind, col_ind):
        dist = cost_point_to_point(pred[row_i], gt[col_i])
        if dist < thresh:
            l_tp.append(pred[row_i])
            matching.append((pred[row_i], gt[col_i]))
            break
        else:
            l_fp.append(pred[row_i])
            l_fn.append(gt[col_i])

    for g_i in range(len(gt)):
        if g_i not in col_ind:
            l_fn.append(gt[g_i])

    print("tp: {}, fp: {}, fn: {}".format(len(l_tp), len(l_fp), len(l_fn)))

    return l_tp, l_fp, l_fn, matching


# get recall and precision
def get_recall_precision(tp, fp, fn):
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)

    return recall, precision


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

    infer(image_paths, model, postprocessors, device, args.output_dir)
