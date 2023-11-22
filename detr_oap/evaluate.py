import torch
from tqdm import tqdm

from datasets.pole import make_Pole_transforms
from evaluation.utils import *

"""
This file will be used to evaluate the model on the validation set
during training. It will be called after each epoch.
"""


@torch.no_grad()
def evaluate(model, dataloader, device, thresh_score, thresh_dist):
    total_tp, total_fp, total_fn = 0, 0, 0
    n_pairwise_matches = 0
    error_sum_x = 0

    model.eval()

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
        keep = probas.max(-1).values >= thresh_score

        # Scale keypoints to the original image size
        points_scaled = rescale_prediction(outputs["pred_boxes"][0, keep], samples.size)
        probas = probas[keep].cpu().data.numpy()

        gt_data = targets["boxes"].cpu().data.numpy()[:, :2]
        l_tp, l_fp, l_fn, matching = get_tp_fp_fn(
            points_scaled, probas, gt_data, thresh_dist
        )

        n_pairwise_matches += len(matching)
        err_x, _, _ = get_err_sum(matching)
        error_sum_x += err_x

        total_tp += len(l_tp)
        total_fp += len(l_fp)
        total_fn += len(l_fn)

    # compute precision and recall
    recall, precision = get_recall_precision(total_tp, total_fp, total_fn)
    if n_pairwise_matches == 0:
        MAE_x = np.inf
    else:
        MAE_x = error_sum_x / n_pairwise_matches

    metrics = {"recall": recall, "precision": precision, "MAE_x": MAE_x}
    return metrics


@torch.no_grad()
def evaluate_val(model, dataloader, device, thresh):
    model.eval()

    tab_probas_points_imgnb = np.empty((0, 4))
    tab_gt_imgnb = np.empty((0, 3))

    print("Inference start...")
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
        keep = probas.max(-1).values > thresh

        # Scale keypoints to the original image size
        points_scaled = rescale_prediction(outputs["pred_boxes"][0, keep], samples.size)
        probas = probas[keep].cpu().data.numpy()

        gt_data = targets["boxes"].cpu().data.numpy()[:, :2]
        img_id = targets["image_id"].cpu().data.numpy()[0]

        if gt_data.shape[0] != 0:
            tab_img_id = img_id * np.ones((gt_data.shape[0], 1))
            tab_gt_imgnb = np.vstack((tab_gt_imgnb, np.hstack((gt_data, tab_img_id))))

        if probas.shape[0] == 0:
            continue

        probas_points = np.hstack((probas, points_scaled))
        tab_img_id = img_id * np.ones((probas_points.shape[0], 1))
        probas_points_imgnb = np.hstack((probas_points, tab_img_id))
        tab_probas_points_imgnb = np.vstack(
            (tab_probas_points_imgnb, probas_points_imgnb)
        )
    print("Inference ended")

    probas_unique = np.unique(tab_probas_points_imgnb[:, 0])
    probas_ordered = np.sort(probas_unique)[::-1]

    tab_all_metrics = np.empty((0, 9))

    print("Start metrics calculation...")
    for proba in tqdm(probas_ordered):
        metrics_for_a_score = get_metrics_for_a_score(
            tab_probas_points_imgnb, tab_gt_imgnb, proba
        )
        tab_all_metrics = np.vstack((tab_all_metrics, metrics_for_a_score))
    print("End calculation metrics")

    return tab_all_metrics


def get_metrics_for_a_score(
    tab_probas_points_imgnb, tab_gt_imgnb, proba, dist_thresh=10
):
    # keep only the points with proba >= proba
    keep = tab_probas_points_imgnb[:, 0] >= proba
    tab_probas_points_imgnb = tab_probas_points_imgnb[keep]
    # get image ids from tab_probas_points_imgnb
    img_ids = np.unique(tab_probas_points_imgnb[:, -1])

    total_tp, total_fp, total_fn = 0, 0, 0
    total_error_sum_x, total_error_sum_l1, total_error_sum_l2 = 0, 0, 0

    for img_id in img_ids:
        keep = tab_probas_points_imgnb[:, -1] == img_id
        probas_points_imgnb = tab_probas_points_imgnb[keep]
        keep = tab_gt_imgnb[:, -1] == img_id
        gt_imgnb = tab_gt_imgnb[keep]

        probas = probas_points_imgnb[:, 0]
        pred = probas_points_imgnb[:, 1:3]
        gt = gt_imgnb[:, :2]

        tp, fp, fn, error_sum_x, error_sum_l1, error_sum_l2 = get_tp_fp_fn_opti(
            pred, probas, gt, dist_thresh
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_error_sum_x += error_sum_x
        total_error_sum_l1 += error_sum_l1
        total_error_sum_l2 += error_sum_l2

    if total_tp == 0:
        MAE_x = np.inf
        MAE_l1 = np.inf
        MAE_l2 = np.inf
    else:
        MAE_x = total_error_sum_x / total_tp
        MAE_l1 = total_error_sum_l1 / total_tp
        MAE_l2 = total_error_sum_l2 / total_tp

    # some fn could have been skipped because there was no pred for an image with gt
    adjuste_fn = tab_gt_imgnb.shape[0] - (total_fn + total_tp)
    total_fn += adjuste_fn
    recall, precision = get_recall_precision(total_tp, total_fp, total_fn)
    tab = np.array(
        [
            proba,
            recall,
            precision,
            total_tp,
            total_fp,
            total_fn,
            MAE_x,
            MAE_l1,
            MAE_l2,
        ]
    )
    return tab
