import csv
import glob
import os
import shutil

from PIL import Image
from sahi.utils.coco import Coco, CocoAnnotation, CocoCategory, CocoImage
from sahi.utils.file import save_json

box_w, box_h = 200, 200
train_per = 0.8
data_dir = "data_manual_annotations"
anotation_dir = "annotations_phx"
img_dir = "images"


data_train_path= os.path.join(data_dir, img_dir,"train")
data_val_path=os.path.join(data_dir, img_dir,"val")
os.makedirs(data_train_path, exist_ok=True)
os.makedirs(data_val_path, exist_ok=True)

# Init coco object
coco_train = Coco()
coco_val = Coco()

# Add categories
coco_train.add_category(CocoCategory(id=0, name="pole"))
coco_val.add_category(CocoCategory(id=0, name="pole"))


anotation_path = os.path.join(data_dir, anotation_dir, "*.csv")

csv_files = glob.glob(anotation_path)

num_train_data = int(len(csv_files) * train_per)
count = 1

# Loop through each CSV file and append its data to the merged_data DataFrame
for csv_file in csv_files:
    with open(csv_file, "r") as csvfile:
        reader = csv.DictReader(csvfile)

        try:
            next(reader)
        except StopIteration:
            pass

        name_img = os.path.basename(csv_file)[:-3] + "jpg"
        filename = os.path.join(data_dir, img_dir, name_img)

        
        height, width = Image.open(filename).size
        if count <= num_train_data:
            path=f"{data_dir}/{img_dir}/train/{name_img}"
        else:
            path=f"{data_dir}/{img_dir}/val/{name_img}"
        
        coco_image = CocoImage(
            file_name=path, height=height, width=width
        )

        for row in reader:
            x = int(float(row["x"]))
            y = int(float(row["y"]))

            xtl = max(0, x - box_w // 2)
            ytl = max(0, y + box_h // 2)
            coco_image.add_annotation(
                CocoAnnotation(bbox=[xtl, ytl, box_w, box_h], category_id=0, category_name="pole")
            )

    if count <= num_train_data:
        coco_train.add_image(coco_image)
        shutil.copy2(filename, data_train_path)
    else:
        coco_val.add_image(coco_image)
        shutil.copy2(filename, data_val_path)
    count += 1


save_json(data=coco_train.json, save_path="train.json")
save_json(data=coco_val.json, save_path="val.json")
