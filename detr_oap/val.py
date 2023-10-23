# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# python val.py --data_path ../data_manual_annotations/images/val --resume PATH_TO_MODEL --device cpu --backbone resnet101 &

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
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import auc
import torch
import util.misc as utils
from datasets.pole import make_Pole_transforms
from models import build_model
from PIL import Image
from scipy.optimize import linear_sum_assignment


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

    parser.add_argument("--thresh", default=0.2, type=float)

    parser.add_argument("--show", default=False, type=bool)

    return parser



def rescale_prediction(out_bbox, size):
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


def cost_point_to_point(pred, gt):
    return np.linalg.norm(pred - gt)


def get_tab_metrics_for_I(pred, probas, gt, thresh):
    tab_metrics = np.empty((0, 7))
    pred = pred.numpy()
    idx = np.argsort(probas.flatten())[::-1]
    pred = pred[idx]
    probas = probas[idx]

    if gt.shape[0] != 0:
        tab = np.array([1, 0, 0, gt.shape[0], 0, 0, 0]).reshape((1, 7))
        tab_metrics = np.vstack((tab_metrics, tab))

    for i, proba in enumerate(probas):
        tp, fp, fn, error_sum_x, error_sum_l1, error_sum_l2 = get_tp_fp_fn_err(
            pred[: i + 1, :], gt, thresh
        )
        tab = np.array([proba[0], tp, fp, fn, error_sum_x, error_sum_l1, error_sum_l2])
        tab = tab.reshape((1, 7))
        tab_metrics = np.vstack((tab_metrics, tab))

    tab_metrics = diff_tab_metrics(tab_metrics)

    return tab_metrics


def diff_tab_metrics(tab_metrics):
    copy_tab_metrics = np.copy(tab_metrics)
    for i in range(1, tab_metrics.shape[0]):
        copy_tab_metrics[i, 1:] = tab_metrics[i, 1:] - tab_metrics[i - 1, 1:]
    return copy_tab_metrics


def hungarian_matching(pred, gt, tresh):
    """Hungarian matching algorithm"""
    # compute the cost matrix
    cost_matrix = np.zeros((len(pred), len(gt)))

    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            if cost_point_to_point(p, g) < tresh:
                cost_matrix[i, j] = -1
            else:
                cost_matrix[i, j] = 0

    # apply the hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    return row_ind, col_ind


# function that return number of true positives, false positives and false negatives
def get_tp_fp_fn_err(pred, gt, thresh):
    tp, fp, fn = 0, 0, 0
    error_sum_x = 0
    error_sum_l1 = 0
    error_sum_l2 = 0

    if gt.shape[0] == 0:
        return 0, pred.shape[0], 0, 0, 0, 0

    pred_ind, gt_ind = hungarian_matching(pred, gt, thresh)

    for p_ind, g_ind in zip(pred_ind, gt_ind):
        dist = cost_point_to_point(pred[p_ind], gt[g_ind])
        if dist < thresh:
            tp += 1
            error_sum_x += abs(pred[p_ind][0] - gt[g_ind][0])
            error_sum_l1 += np.linalg.norm(pred[p_ind] - gt[g_ind], ord=1)
            error_sum_l2 += np.linalg.norm(pred[p_ind] - gt[g_ind], ord=2)
        else:
            fp += 1
    fn = gt.shape[0] - tp

    return tp, fp, fn, error_sum_x, error_sum_l1, error_sum_l2


# get recall and precision
def get_recall_precision(tp, fp, fn):
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)

    return recall, precision


def get_tab_metrics_for_all_I(tab):
    idx = np.argsort(tab[:, 0])[::-1]
    tab = tab[idx]

    tab_with_unique_score = tab[0, :].reshape((1, 7))
    ind = 0
    for i in range(1, tab.shape[0]):
        if tab[i, 0] != tab[i - 1, 0]:
            tab_with_unique_score = np.vstack((tab_with_unique_score, tab[i, :]))
            ind += 1
        else:
            tab_with_unique_score[ind, :] = tab_with_unique_score[ind, :] + tab[i, :]
            tab_with_unique_score[ind, 0] = tab[i, 0]

    tab_cummulate = tab_with_unique_score
    for i in range(1, tab_cummulate.shape[0]):
        tab_cummulate[i, 1:] = tab_cummulate[i, 1:] + tab_cummulate[i - 1, 1:]

    tab_with_precision_and_recall = np.empty((0, 6))
    for i in range(tab_cummulate.shape[0]):
        tab_cummulate[i, 4] = tab_cummulate[i, 4] / tab_cummulate[i, 1]
        tab_cummulate[i, 5] = tab_cummulate[i, 5] / tab_cummulate[i, 1]
        tab_cummulate[i, 6] = tab_cummulate[i, 6] / tab_cummulate[i, 1]
        recall, precision = get_recall_precision(
            tab_cummulate[i, 1], tab_cummulate[i, 2], tab_cummulate[i, 3]
        )
        t = np.array([tab_cummulate[i, 0], precision, recall])
        t = np.hstack((t, tab_cummulate[i, 4:7]))
        tab_with_precision_and_recall = np.vstack((tab_with_precision_and_recall, t))

    return tab_with_precision_and_recall


def plot_AP_curve(tab):
    fig, ax = plt.subplots()
    ax.plot(tab[:, 2], tab[:, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("AP curve")
    plt.show()


@torch.no_grad()
def infer(images_path, model, postprocessors, device, output_path):
    # load grount truth json from data_manual_annotations/val.json
    gt = "../data_manual_annotations/annotations_tx_reviewed_final"

    model.eval()
    duration = 0

    n_pairwise_matches = 0
    tab_all_metrics = np.empty((0, 7))

    for i, img_sample in tqdm(enumerate(images_path)):
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
        keep = probas.max(-1).values > args.thresh

        # Scale bbox to the same size as the original
        bboxes_scaled = rescale_prediction(outputs["pred_boxes"][0, keep], orig_image.size)
        probas = probas[keep].cpu().data.numpy()


        print("Start matching")
        if bboxes_scaled.shape[0] == 0 and gt_data.shape[0] == 0:
            continue
        tab_metrics = get_tab_metrics_for_I(bboxes_scaled, probas, gt_data, thresh=10)
        print("End matching")

        tab_all_metrics = np.vstack((tab_all_metrics, tab_metrics))

        n_pairwise_matches += tab_metrics.shape[0]
        print(f"End")

    # compute precision and recall
    tab_all_metrics = get_tab_metrics_for_all_I(tab_all_metrics)
    plot_AP_curve(tab_all_metrics)
    print(f"score, precision, recall, MAEx, MAE_L1, MAE_L2:")
    print(f"tab_all_metrics: {tab_all_metrics}")
    t = np.array((tab_all_metrics[:,1], tab_all_metrics[:,2]))
    idx = np.argsort(t[:,0].flatten())
    t = t[idx]

    AUC = auc(t[0], t[1])
    print(f"AUC: {AUC}")

    avg_duration = duration / len(images_path)
    print("Avg. Time: {:.3f}s".format(avg_duration))




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
