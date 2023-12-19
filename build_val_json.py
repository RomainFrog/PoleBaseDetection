import csv
import glob
import os
import shutil

from PIL import Image
from sahi.utils.coco import Coco, CocoAnnotation, CocoCategory, CocoImage
from sahi.utils.file import save_json
import argparse
from tqdm import tqdm
import uuid

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data_manual_annotations", help="data directory")
parser.add_argument("--output_dir", default="datasets", help="output dataset directory")
parser.add_argument("--output_name", default="val", help="output json name")
parser.add_argument("--size", default=50, help="size of the bounding box")
args = parser.parse_args()



data_dir = args.data_dir
size = int(args.size)
output_dir = args.output_dir
images = glob.glob(os.path.join(data_dir, 'images', "*.jpg"))

coco_set = Coco()


for img_file in tqdm(images):
    height, width = Image.open(img_file).size
    name_img = os.path.basename(img_file)
    coco_image = CocoImage(file_name=f"{data_dir}/images/{name_img}", height=height, width=width)

    with open(os.path.join(os.path.join(data_dir, 'annotations'), name_img[:-3] + "csv"), "r") as csvfile:
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
        coco_set.add_image(coco_image)


coco_path= os.path.join(output_dir, args.output_name + ".json")
save_json(data=coco_set.json, save_path=coco_path)
