import math
import os
import sys
from typing import Iterable

import torch

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    for samples, targets in data_loader:
        samples = samples.to(device)
        outputs = model(samples)
        print(outputs)
        print(targets)


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    pass