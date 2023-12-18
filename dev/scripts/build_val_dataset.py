"""
This script allows the user to build a validation dataset using a .csv file listing
the basenames of the images to be used for validation. The script will then create a
val.json file using sahi.
"""

import os
import shutil
import pandas as pd
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Build a validation dataset using a .csv file listing the basenames of the images to be used for validation.')
parser.add_argument('--img_src_folder', type=str, default='../../data_manual_annotations/images/val', help='Path to the image source folder.')
parser.add_argument('--ann_src_folder', type=str, default='../../data_manual_annotations/final_dataset', help='Path to the annotation source folder.')
parser.add_argument('--gt_file', type=str, default='../data/gt_data.csv', help='Path to the gt_data.csv file.')
parser.add_argument('--density_percentile', type=float, default=5, help='The percentile of the density to be used for validation. Default is 10.')
parser.add_argument('--y_gt_percentile', type=float, default=15, help='The percentile of the y_gt to be used for validation. Default is 10.')
parser.add_argument('--output_dir', type=str, default='../../datasets', help='Path to the output folder.')



def build_density_df(gt):
    rows = []
    for index,row in gt.iterrows():
        rows.append({'basename': row['basename'], 'gt_count': len(gt[gt.basename == row['basename']])})
    gt_context = pd.DataFrame(rows)
    gt_context.drop_duplicates(inplace=True)

    return gt_context

def build_avg_y_df(gt):
    # build a new df based on gt and get the avg y_gt for each basename
    rows = []
    for index, row in gt.iterrows():
        rows.append({'basename': row['basename'], 'y_gt': row['y_gt']})

    gt_avg = pd.DataFrame(rows)
    gt_avg.drop_duplicates(inplace=True)
    gt_avg['y_gt_avg'] = gt_avg.apply(lambda row: gt_avg[gt_avg.basename == row['basename']]['y_gt'].median(), axis=1)
    gt_avg.drop(columns=['y_gt'], inplace=True)
    gt_avg.drop_duplicates(inplace=True)
    return gt_avg



def main():
    args = parser.parse_args()

    try:
        gt = pd.read_csv(args.gt_file)
        gt.rename(columns={'# basename': 'basename'}, inplace=True)
    except:
        print("No gt_data.csv file found. Please run build_gt_data.py first.")
        exit()
    else:
        print("gt_data.csv file found. Ground Truth data loaded!")

    print("Building density dataframe...")
    gt_density = build_density_df(gt)

    print("Building avg_y dataframe...")
    gt_avg_y = build_avg_y_df(gt)


    # Get the basenames of the images to be used for validation
    bottom_perc_avg_y = gt_avg_y[gt_avg_y.y_gt_avg <= gt_avg_y.quantile(float(args.y_gt_percentile)/100)['y_gt_avg']]['basename'].tolist()
    top_perc_avg_y = gt_avg_y[gt_avg_y.y_gt_avg >= gt_avg_y.quantile(1-(float(args.y_gt_percentile)/100))['y_gt_avg']]['basename'].tolist()
    bottom_perc_density = gt_density[gt_density.gt_count <= gt_density.quantile(float(args.density_percentile)/100)['gt_count']]['basename'].tolist()
    top_perc_density = gt_density[gt_density.gt_count >= gt_density.quantile(1-(float(args.density_percentile)/100))['gt_count']]['basename'].tolist()
    

    # Copy the images and annotations
    output_folder = args.output_dir + '_' + f"bottom_{args.y_gt_percentile}_avg_y"
    # Create the output folder
    os.makedirs(os.path.join(output_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'annotations'), exist_ok=True)
    for basename in tqdm(bottom_perc_avg_y):
        # Get the basename of the image and annotation (they should be the same wtihout the extension)
        # Copy the image
        shutil.copy(os.path.join(args.img_src_folder, str(basename) + '.jpg'), os.path.join(output_folder, 'images'))
        # Copy the annotation
        shutil.copy(os.path.join(args.ann_src_folder, str(basename) + '.csv'), os.path.join(output_folder, 'annotations'))

    output_folder = args.output_dir + '_' + f"top_{args.y_gt_percentile}_avg_y"
    # Create the output folder
    os.makedirs(os.path.join(output_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'annotations'), exist_ok=True)
    for basename in tqdm(top_perc_avg_y):
        # Get the basename of the image and annotation (they should be the same wtihout the extension)
        # Copy the image
        shutil.copy(os.path.join(args.img_src_folder, str(basename) + '.jpg'), os.path.join(output_folder, 'images'))
        # Copy the annotation
        shutil.copy(os.path.join(args.ann_src_folder, str(basename) + '.csv'), os.path.join(output_folder, 'annotations'))

    output_folder = args.output_dir + '_' + f"bottom_{args.density_percentile}_density"
    # Create the output folder
    os.makedirs(os.path.join(output_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'annotations'), exist_ok=True)
    for basename in tqdm(bottom_perc_density):
        # Get the basename of the image and annotation (they should be the same wtihout the extension)
        # Copy the image
        shutil.copy(os.path.join(args.img_src_folder, str(basename) + '.jpg'), os.path.join(output_folder, 'images'))
        # Copy the annotation
        shutil.copy(os.path.join(args.ann_src_folder, str(basename) + '.csv'), os.path.join(output_folder, 'annotations'))

    output_folder = args.output_dir + '_' + f"top_{args.density_percentile}_density"
    # Create the output folder
    os.makedirs(os.path.join(output_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'annotations'), exist_ok=True)
    for basename in tqdm(top_perc_density):
        # Get the basename of the image and annotation (they should be the same wtihout the extension)
        # Copy the image
        shutil.copy(os.path.join(args.img_src_folder, str(basename) + '.jpg'), os.path.join(output_folder, 'images'))
        # Copy the annotation
        shutil.copy(os.path.join(args.ann_src_folder, str(basename) + '.csv'), os.path.join(output_folder, 'annotations'))

    print("Done!")


if __name__ == '__main__':
    main()