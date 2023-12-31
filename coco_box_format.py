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
parser.add_argument("--output_dir", default="datasets/default_dataset", help="output dataset directory")
parser.add_argument("--size", default=50, help="size of the bounding box")
parser.add_argument("--bdd100k", action="store_true", help="use bdd100k dataset")
parser.add_argument("--bdd100k_val", action="store_true", help="use bdd100k validation dataset")
args = parser.parse_args()


# def get_fitting_bbox(x_c, y_c, size, width, height):
#     """
#     x_c: x coordinate of the center of the bounding box
#     y_c: y coordinate of the center of the bounding box
#     size: size of the bounding box
#     width: width of the image
#     height: height of the image
    
#     returns: xtl, ytl, bbox_width, bbox_height
    
#     If the bounding box is outside of the image, the function returns the largest possible bbox 
#     that fits in the image but its center is still (x_c, y_c)
#     """

#     # Calculate potential bbox_width and bbox_height without exceeding the remaining width and height
#     bbox_width = min(size, (width - x_c)*2)
#     bbox_height = min(size, (height - y_c)*2)

#     # Calculate potential xtl and ytl without going below 0
#     xtl = max(0, x_c - bbox_width//2)
#     ytl = max(0, y_c - bbox_height//2)

#     return xtl, ytl, bbox_width, bbox_height





data_dir = args.data_dir
bdd100k = args.bdd100k
bdd100k_val = args.bdd100k_val
compi_labels = args.annotation_dir
bdd100k_labels = os.path.join(data_dir, "annotations_bdd100k")
size = int(args.size)
output_dir = args.output_dir

compi_img_dir = "images"
bdd_img_dir = "images_bdd100k"


compi_data_train_path= os.path.join(data_dir, compi_img_dir,"train")
compi_data_val_path=os.path.join(data_dir, compi_img_dir,"val")
if bdd100k:
    bdd100k_data_train_path= os.path.join(data_dir, bdd_img_dir,"train")
if bdd100k_val:
    bdd100k_data_val_path=os.path.join(data_dir, bdd_img_dir,"val") 

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
            xtl = max(0, x - size // 2)
            ytl = max(0, y - size // 2)
            coco_image.add_annotation(
                CocoAnnotation(bbox=[xtl, ytl, size, size], category_id=0, category_name="pole")
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
                x = int(float(row["x"]) * height)
                y = int(float(row["y"]) * width)
                xtl = max(0, x - size // 2)
                ytl = max(0, y - size // 2)
                coco_image.add_annotation(
                    CocoAnnotation(bbox=[xtl, ytl, size, size], category_id=0, category_name="pole")
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
                xtl = max(0, x - size // 2)
                ytl = max(0, y - size // 2)
                coco_image.add_annotation(
                    CocoAnnotation(bbox=[xtl, ytl, size, size], category_id=0, category_name="pole")
                )
            coco_val.add_image(coco_image)

else:
    for img_file in tqdm(bdd100k_val_images):
        height, width = Image.open(img_file).size
        name_img = os.path.basename(img_file)
        coco_image = CocoImage(file_name=f"images_bdd100k/val/{name_img}", height=height, width=width)

        with open(os.path.join(bdd100k_labels, "val", name_img[:-3] + "txt"), "r") as csvfile:
            reader = csv.DictReader(csvfile)

            try:
                next(reader)
            except StopIteration:
                pass

            for row in reader:
                x = int(float(row["x"]) * height)
                y = int(float(row["y"]) * width)
                xtl = max(0, x - size // 2)
                ytl = max(0, y - size // 2)
                coco_image.add_annotation(
                    CocoAnnotation(bbox=[xtl, ytl, size, size], category_id=0, category_name="pole")
                )
            coco_val.add_image(coco_image)


coco_train_path= os.path.join(output_dir, "train.json")
coco_val_path= os.path.join(output_dir, "val.json")
save_json(data=coco_train.json, save_path=coco_train_path)
save_json(data=coco_val.json, save_path=coco_val_path)
