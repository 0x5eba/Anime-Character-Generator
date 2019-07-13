import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from PIL import Image

class AttrDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with featrues.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.attr_list = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.attr_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.attr_list.ix[idx, 0])+".png")
        image = Image.open(img_name).convert('RGB')
        attrs = self.attr_list.ix[idx, 1:].values

        hair = torch.FloatTensor(attrs[0:12])
        eye = torch.FloatTensor(attrs[12:])

        if self.transform:
            image = self.transform(image)

        return image, hair, eye

transform_anime = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

path_data = '../dataset/data/'
A_train_dataset = AttrDataset('./create_data/features.csv', path_data, transform_anime)
train_loader = DataLoader(A_train_dataset, batch_size=64, num_workers=16, shuffle=True, drop_last=True)