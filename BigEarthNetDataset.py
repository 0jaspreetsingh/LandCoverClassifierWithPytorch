from torch.utils.data import Dataset
import cv2
import os
import pandas as pd
import torch
import numpy
import BigEarthNetConfig

class BigEarthNetDataset(Dataset):

    def __init__(self, data_path, label_filename, transform=None, target_transform = None):
        self.fileList = os.listdir(path=data_path)
        labelPath = os.path.join(data_path, label_filename)
        self.img_labels = pd.read_csv(labelPath, header=None)
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.img_labels.index)

    def __getitem__(self, index):
        imagePath = os.path.join(self.data_path,"{}.jpg".format(self.img_labels.iloc[index, 0]))
        image = cv2.imread(imagePath)
        labels = self.img_labels.iloc[index, 1:].to_numpy(dtype=numpy.float32)
        if self.transform:
            image = self.transform(image.astype(numpy.float32))
        labels = torch.tensor(labels)
        return image, labels

    def getName(self, index):
        return self.img_labels.iloc[index, 0]

    def getLables(self, index):
        list = []
        labels = self.img_labels.iloc[index, 1:].to_numpy(dtype=numpy.uint8)
        for index, value in enumerate(labels):
            if(value==1): list.append(BigEarthNetConfig.BigEarthNetLabelNames.get(index))
        return ','.join(list)
