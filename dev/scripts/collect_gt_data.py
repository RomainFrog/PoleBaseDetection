"""
This script is used to collect ground truth data from a folder
and merge them into a single csv file.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def main(gt_folder):
    val_folder = '../../data_manual_annotations/images/val'
    # list all files in the folder if they end with .csv and if the basename is found in the val folder
    val_basenames = [os.path.basename(f).split('.')[0] for f in os.listdir(val_folder) if f.endswith('.jpg')]
    files = [f for f in os.listdir(gt_folder) if f.endswith('.csv') and os.path.basename(f).split('.')[0] in val_basenames]
    # initialize a numpy array of size 0x3
    gt_data = np.zeros((0, 3))
    # loop through all files
    for file in tqdm(files):
        # load the file but only the first two columns
        try:
            data = pd.read_csv(gt_folder + file, usecols=[1,2], skiprows=1, header=None).to_numpy()
        except:
            continue
        else:
            # There can be no data in the csv file
            # add a new column on the left with the basename of the file
            data = np.insert(data, 0, values=0, axis=1)
            data[:,0] = os.path.basename(file).split('.')[0]
            gt_data = np.append(gt_data, data, axis=0)

    # save the gt_data array as a csv file
    np.savetxt('../gt_data.csv', gt_data, delimiter=',', header='basename,x_gt,y_gt', comments='', fmt='%d,%d,%d')
    


if __name__ == '__main__':
    gt_folder = '../../data_manual_annotations/final_dataset/'
    main(gt_folder)