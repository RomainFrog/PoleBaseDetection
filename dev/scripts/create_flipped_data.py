"""
Data augmentations script for flipping images horizontal and adapt
annotations accordingly.
"""

import os
import numpy as np
import glob
import csv
from PIL import Image
from tqdm import tqdm

source_dir = "../../data_manual_annotations/images/train"
target_dir = "../../data_manual_annotations/flipped_images"
source_labels_dir = "../../data_manual_annotations/final_dataset"
target_labels_dir = "../../data_manual_annotations/final_dataset_flipped"

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

if not os.path.exists(target_labels_dir):
    os.makedirs(target_labels_dir)

images = glob.glob(os.path.join(source_dir, "*.jpg"))


for img_file in tqdm(images):
    name_img = os.path.basename(img_file)
    img = Image.open(img_file)
    img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
    img_flip.save(os.path.join(target_dir, name_img))
    # get width and height of image
    width, height = img.size

    with open(os.path.join(source_labels_dir, name_img[:-3] + "csv"), "r") as csvfile:
        with open(os.path.join(target_labels_dir, name_img[:-3] + "csv"), "w") as target_csv:
            target_csv.write(f",x,y,category\n")
            reader = csv.DictReader(csvfile)
            try:
                next(reader)
            except StopIteration:
                pass

            # iter with enumerate
            for i, row in enumerate(reader):
                x = int(float(row["x"]))
                y = int(float(row["y"]))
                # flip coordinates horizontally
                x = width - x

                target_csv.write(f"{i},{x},{y},pole\n")