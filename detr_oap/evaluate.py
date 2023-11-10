import torch
from evaluation.utils import *
from datasets.pole import make_Pole_transforms


"""
This file will be used to evaluate the model on the validation set
during training. It will be called after each epoch.

The function will compute the following metrics:
- Recall
- Precision
- F1 score
- AUC (later)
- MAE in x
"""

@torch.no_grad()
def evaluate(model, dataloader, device):

    total_tp, total_fp, total_fn = 0, 0, 0
    n_pairwise_matches = 0
    error_sum_x = 0

    model.eval()
    for samples, targets in dataloader:
        image = samples[0]
        # /!\ targets is still in shape of bbox and we only need to keep
        # the first two coordinates (x, y)$
        h, w = image.shape[1:]
        print(f"image_width = {w},image height = {h}")
        # transform = make_Pole_transforms("val")
        # dummy_target = {
        #     "size": torch.as_tensor([int(h), int(w)]),
        #     "orig_size": torch.as_tensor([int(h), int(w)]),
        # }
        # image, _ = transform(samples, dummy_target)
        # image = image.unsqueeze(0).to(device)


        outputs = model(samples)

        outputs["pred_logits"] = outputs["pred_logits"].cpu()
        outputs["pred_boxes"] = outputs["pred_boxes"].cpu()


        probas = outputs["pred_logits"].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.85

        # Scale keypoints to the original image size
        points_scaled = rescale_prediction(outputs["pred_boxes"][0, keep], (w,h))
        probas = probas[keep].cpu().data.numpy()

        gt_data = targets['boxes'].cpu().data.numpy()[:,:2]
        print(points_scaled)
        print(gt_data)

        l_tp, l_fp, l_fn, matching = get_tp_fp_fn(
            points_scaled, probas, gt_data, 10, hungarian_matching
        )

        n_pairwise_matches += len(matching)
        err_x, _, _ = get_err_sum(matching)
        error_sum_x += err_x

        total_tp += len(l_tp)
        total_fp += len(l_fp)
        total_fn += len(l_fn)


    # compute precision and recall
    recall, precision = get_recall_precision(total_tp, total_fp, total_fn)
    MAE_x = error_sum_x / n_pairwise_matches

    print("Recall: {:.3f}, Precision: {:.3f}".format(recall, precision))
    print("MAE_x: {:.3f}".format(MAE_x))