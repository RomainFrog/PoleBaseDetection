import argparse
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

from detr.datasets.pole import PoleDetection

# Load dataset
img_dir = "data_manual_annotations/images"
ann_dir = "data_manual_annotations/train.json"

dataset = PoleDetection(img_dir, ann_dir, transforms=None, return_masks=False)

# Create a cv2 window to display images from val.json
cv2.namedWindow("Pole Detection", cv2.WINDOW_NORMAL)

# Display images from val.json
for i in range(len(dataset)):
    img, target = dataset[i]
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # display bounding boxes on top of the image
    for box in target["boxes"]:
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(x1, y1, x2, y2)
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Pole Detection", img)
    # if key pressed is 'q' then exit the loop
    if cv2.waitKey(0) == ord("q"):
        break
    cv2.waitKey(0)


cv2.destroyAllWindows()
