import pandas as pd
import os
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils import data

def getData(mode):
    if mode == 'train':
        df = pd.read_csv('dataset/train.csv')
        path = df['filepaths'].tolist()
        label = df['label_id'].tolist()
        return path, label
    elif mode == 'valid':
        df = pd.read_csv('dataset/valid.csv')
        path = df['filepaths'].tolist()
        label = df['label_id'].tolist()
        return path, label
    elif mode == 'test':
        df = pd.read_csv('dataset/test.csv')
        path = df['filepaths'].tolist()
        label = df['label_id'].tolist()
        return path, label
    
        
class BufferflyMothLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        print("> Found %d images..." % (len(self.img_name)))  

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        imag_path = os.path.join(self.root, self.img_name[index])
        img = Image.open(imag_path)
        # img = np.asarray(img)
        label = self.label[index]
        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.ColorJitter(brightness=(0.5, 1.5)), 
                transforms.RandomAffine(degrees=0, scale=(0.5, 1.0)), 
                transforms.Resize(300),  
                transforms.RandomCrop(224),  
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ])
            img = self.transform(img)
        elif self.mode == 'valid' or self.mode == 'test':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ])
            img = self.transform(img)
        return img, label
if __name__ == "__main__":
    train_dataset = BufferflyMothLoader(root='dataset', mode='train')

    img, label = train_dataset[0]
    print(img, label)