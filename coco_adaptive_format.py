import csv
import glob
import os
import shutil
import random
from PIL import Image
from sahi.utils.coco import Coco, CocoAnnotation, CocoCategory, CocoImage
from sahi.utils.file import save_json
import argparse
from tqdm import tqdm

def depth_regression(X):
    return 1221.1025860930577 -6.13183029e+00*X + 1.03173169e-02*X**2 -5.76531414e-06*X**3

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data_manual_annotations", help="data directory")
    parser.add_argument("--anotation_dir", default="annotations_tx_reviewed_final", help="annotation directory")
    parser.add_argument("--output_dir", default="default_dataset", help="output dataset directory")
    parser.add_argument("--size", default=200, help="size of the bounding box")
    parser.add_argument
    args = parser.parse_args()

    data_dir = args.data_dir
    anotation_dir = args.anotation_dir
    output_dir = args.output_dir
    size = args.size

    random.seed(42)
    train_per = 0.8
    alpha = 10
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
    random.shuffle(csv_files)

    num_train_data = int(len(csv_files) * train_per)
    print(num_train_data)
    count = 1

    # Loop through each CSV file and append its data to the merged_data DataFrame
    for csv_file in tqdm(csv_files):
        with open(csv_file, "r") as csvfile:
            reader = csv.DictReader(csvfile)

            try:
                next(reader)
            except StopIteration:
                pass

            name_img = os.path.basename(csv_file)[:-3] + "jpg"
            filename = os.path.join(data_dir, img_dir, name_img)

            if not os.path.exists(filename):
                continue
            height, width = Image.open(filename).size
            if count <= num_train_data:
                path=f"train/{name_img}"
            else:
                path=f"val/{name_img}"
            
            coco_image = CocoImage(
                file_name=path, height=height, width=width
            )

            for row in reader:
                x = int(float(row["x"]))
                y = int(float(row["y"]))

                
                d = depth_regression(y)
                box_w = max(5,size//d * alpha)
                print(box_w)
                xtl = max(0, x - box_w // 2)
                ytl = max(0, y - box_w // 2)
                coco_image.add_annotation(
                    CocoAnnotation(bbox=[xtl, ytl, box_w, box_w], category_id=0, category_name="pole")
                )

        if count <= num_train_data:
            coco_train.add_image(coco_image)
            shutil.copy2(filename, data_train_path)
        else:
            coco_val.add_image(coco_image)
            shutil.copy2(filename, data_val_path)
        count += 1


    save_json(data=coco_train.json, save_path="data_manual_annotations/train.json")
    save_json(data=coco_val.json, save_path="data_manual_annotations/val.json")


if __name__ == "__main__":
    main()