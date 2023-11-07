import torch
from evaluation.utils import *


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

    model.eval()

    for samples, targets in dataloader:
        # /!\ targets is still in shape of bbox and we only need to keep
        # the first two coordinates (x, y)
        pass

