import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import pandas as pd
from utils import *
import utils
import imageio as io
from PIL import Image
'''
def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]
'''
class Custom_DataLoader(Dataset):
    def __init__(self, root_dir, labels, transform = None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(labels, sep=' ', header=None, names=['Class', 'Label'])
        self.transform = transform

    def __getitem__(self, index):
        image_name = os.path.join(self.root_dir, self.annotations['Class'][index])
        image = Image.open(image_name)
        
        if self.transform:
            image = self.transform(image)
        label = self.annotations['Label'][index]
        return image, label

    def __len__(self):
        return len(self.annotations)

def load_syn2real_data(path, label_file, shuffle = True, batch_size = 64):
    transform = transforms.Compose([
                transforms.Resize((64, 64)),       
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                ])
    data = Custom_DataLoader(path, label_file, transform)
    data_loader = DataLoader(data, shuffle = shuffle, batch_size = batch_size)
    
    return data_loader 
