import csv
import glob
import os
import shutil

from PIL import Image
from sahi.utils.coco import Coco, CocoAnnotation, CocoCategory, CocoImage
from sahi.utils.file import save_json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data_manual_annotations", help="data directory")
parser.add_argument("--annotation_dir", default="final_dataset", help="annotation directory")
parser.add_argument("--bdd100k", action="store_true", help="use bdd100k dataset")
parser.add_argument("--bdd100k_val", action="store_true", help="use bdd100k validation dataset")
args = parser.parse_args()

data_dir = args.data_dir
bdd100k = args.bdd100k
bdd100k_val = args.bdd100k_val
compi_labels = args.annotation_dir
bdd100k_labels = os.path.join(data_dir, "annotations_bdd100k")

compi_img_dir = "images"
bdd_img_dir = "images_bdd100k"


compi_data_train_path= os.path.join(data_dir, compi_img_dir,"train")
compi_data_val_path=os.path.join(data_dir, compi_img_dir,"val")
if bdd100k:
    bdd100k_data_train_path= os.path.join(data_dir, bdd_img_dir,"train")
if bdd100k_val:
    bdd100k_data_val_path=os.path.join(data_dir, bdd_img_dir,"valid") 

# Init coco object
coco_train = Coco()
coco_val = Coco()
# Add categories
coco_train.add_category(CocoCategory(id=0, name="pole"))
coco_val.add_category(CocoCategory(id=0, name="pole"))

compi_labels_path = os.path.join(data_dir, compi_labels)

compi_train_images = glob.glob(os.path.join(compi_data_train_path, "*.jpg"))
compi_val_images = glob.glob(os.path.join(compi_data_val_path, "*.jpg"))
if bdd100k:
    bdd100k_train_images = glob.glob(os.path.join(bdd100k_data_train_path, "*.jpg"))
if bdd100k_val:
    bdd100k_val_images = glob.glob(os.path.join(bdd100k_data_val_path, "*.jpg"))

# Loop through each image file and append its data to the merged_data DataFrame
for img_file in tqdm(compi_train_images):
    height, width = Image.open(img_file).size
    name_img = os.path.basename(img_file)
    coco_image = CocoImage(file_name=f"images/train/{name_img}", height=height, width=width)

    with open(os.path.join(compi_labels_path, name_img[:-3] + "csv"), "r") as csvfile:
        reader = csv.DictReader(csvfile)

        try:
            next(reader)
        except StopIteration:
            pass

        for row in reader:
            x = int(float(row["x"]))
            y = int(float(row["y"]))
            coco_image.add_annotation(
                CocoAnnotation(bbox=[x, y, 1, 1], category_id=0, category_name="pole")
            )
        coco_train.add_image(coco_image)


if bdd100k:
    for img_file in tqdm(bdd100k_train_images):
        height, width = Image.open(img_file).size
        name_img = os.path.basename(img_file)

        coco_image = CocoImage(file_name=f"images_bdd100k/train/{name_img}", height=height, width=width)

        with open(os.path.join(bdd100k_labels,"train", name_img[:-3] + "txt"), "r") as csvfile:
            reader = csv.DictReader(csvfile)

            try:
                next(reader)
            except StopIteration:
                pass

            for row in reader:
                x = int(float(row["x"]) * width)
                y = int(float(row["y"]) * height)
                coco_image.add_annotation(
                    CocoAnnotation(bbox=[x, y, 1, 1], category_id=0, category_name="pole")
                )
            coco_train.add_image(coco_image)


if not bdd100k_val:

    for img_file in tqdm(compi_val_images):
        height, width = Image.open(img_file).size
        name_img = os.path.basename(img_file)
        coco_image = CocoImage(file_name=f"images/val/{name_img}", height=height, width=width)

        with open(os.path.join(compi_labels_path, name_img[:-3] + "csv"), "r") as csvfile:
            reader = csv.DictReader(csvfile)

            try:
                next(reader)
            except StopIteration:
                pass

            for row in reader:
                x = int(float(row["x"]))
                y = int(float(row["y"]))
                coco_image.add_annotation(
                    CocoAnnotation(bbox=[x, y, 1, 1], category_id=0, category_name="pole")
                )
            coco_val.add_image(coco_image)

else:
    for img_file in tqdm(bdd100k_val_images):
        height, width = Image.open(img_file).size
        name_img = os.path.basename(img_file)
        coco_image = CocoImage(file_name=f"images_bdd100k/val/{name_img}", height=height, width=width)

        with open(os.path.join(bdd100k_labels, "valid", name_img[:-3] + "txt"), "r") as csvfile:
            reader = csv.DictReader(csvfile)

            try:
                next(reader)
            except StopIteration:
                pass

            for row in reader:
                x = int(float(row["x"]) * width)
                y = int(float(row["y"]) * height)
                coco_image.add_annotation(
                    CocoAnnotation(bbox=[x, y, 1, 1], category_id=0, category_name="pole")
                )
            coco_val.add_image(coco_image)


save_json(data=coco_train.json, save_path="data_manual_annotations/train.json")
save_json(data=coco_val.json, save_path="data_manual_annotations/val.json")
