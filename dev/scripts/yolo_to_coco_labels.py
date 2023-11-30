"""
Script that adds a header to each csv files in a directory
"""
import os

from tqdm import tqdm


import os

def convert_txt_files(folder_path):
    header = "idx,x,y,width,height"

    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Insert header at the beginning
            lines.insert(0, header + '\n')

            # Convert space-separated values to comma-separated values
            lines = [line.strip().replace(' ', ',') + '\n' for line in lines]

            with open(file_path, 'w') as file:
                file.writelines(lines)

if __name__ == "__main__":
    folder_path = "data_manual_annotations/annotations_bdd100k/valid"  # Replace with the actual path to your folder
    convert_txt_files(folder_path)


