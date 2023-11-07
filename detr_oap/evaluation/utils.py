import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

def rescale_prediction(out_bbox, size):
    img_w, img_h = size
    b = out_bbox * torch.tensor([img_w, img_h], dtype=torch.float32)
    return b


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


def hungarian_matching(pred, gt, thresh):
    """Hungarian matching algorithm"""
    # compute the cost matrix
    cost_matrix = np.zeros((len(pred), len(gt)))

    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            dist = np.linalg.norm(p - g, ord=2)
            if dist < thresh:
                if dist == 0:
                    cost_matrix[i, j] = -999999999
                else:
                    cost_matrix[i, j] = -1 / dist
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
            dist = np.linalg.norm(p - g, ord=2)
            if dist <= thresh:
                row_ind.append(np.where(pred == p)[0][0])
                col_ind.append(np.where(gt == g)[0][0])
                gt_copy = np.delete(gt_copy, np.where(gt_copy == g)[0], axis=0)
                break

        row_ind.append(np.where(pred == p)[0][0])
        col_ind.append(-1)

    return row_ind, col_ind


def get_tp_fp_fn(pred, probas, gt, dist_thresh, matching_func=hungarian_matching):
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
        elif np.linalg.norm(pred[row_i] - gt[col_i], ord=2) <= dist_thresh:
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