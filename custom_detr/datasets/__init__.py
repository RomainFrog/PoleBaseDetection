import torch.utils.data
import torchvision

from .pole import build_pole

def build_dataset(image_set, args):
    return build_pole(image_set, args)
