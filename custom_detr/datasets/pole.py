"""
Dataloader for pole dataset
"""
import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms as T

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class PoleDataset(torch.utils.data.Dataset):
    
    def __init__(self, root_dir, image_set='train', transform=transform):
        """
        Args:
            root_dir (str): Directory with all the images and CSV files.
            transform (callable, optional): Optional transform to be applied to the image.
        """
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, 'images', image_set)
        self.annotations_dir = os.path.join(root_dir, 'annotations', image_set)
        self.transform = transform
        self.image_files = [f for f in os.listdir(self.images_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.image_files[idx])
        image = Image.open(img_name)
        basename = os.path.basename(img_name)
        # read annotations from csv file
        csv_name = os.path.join(self.annotations_dir, basename.replace('.jpg', '.csv'))
        annotations = np.loadtxt(csv_name, delimiter=',', skiprows=1, usecols=(1,2))
        # if there is only one annotation, convert it to a 2D array
        if len(annotations.shape) == 1:
            annotations = annotations.reshape(1,2)
        
        # create empty tensor of size (100,2)
        coords = torch.zeros((100,2))

        # fill in the tensor with the annotations
        if annotations.shape[0] != 0:
            coords[:annotations.shape[0], :] = torch.tensor(annotations)

        if self.transform:
            image = self.transform(image)

        # convert image to tensor
        image = torch.tensor(np.array(image))
        return image, coords



def build_pole(image_set, args):
    return PoleDataset(args.data_path, image_set)