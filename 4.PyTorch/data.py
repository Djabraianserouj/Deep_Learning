from importlib_metadata import csv
from sqlalchemy import Float
from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    def __init__(self, data, mode):
        self.data = data 
        self.mode = mode  #mode can be either val or train of dtype=string
        if self.mode == "train":
            self._transform = tv.transforms.Compose([tv.transforms.ToPILImage(), 
                                                tv.transforms.ToTensor(), 
                                                tv.transforms.RandomVerticalFlip(p=0.5),
                                                tv.transforms.Normalize(train_mean, train_std)])
        else:
            self._transform = tv.transforms.Compose([tv.transforms.ToPILImage(), 
                                                tv.transforms.ToTensor(), 
                                                tv.transforms.Normalize(train_mean, train_std)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        self.index = index  
        csv_path = self.data.iloc[index,0]

        image = imread(csv_path)
        image = gray2rgb(image)
        image = self._transform(image)
        image = torch.tensor(image)

        labels = self.data.iloc[index,1:]
        labels = torch.FloatTensor(labels)
        sample = (image, labels)
        return sample


