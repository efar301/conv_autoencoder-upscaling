import os
from PIL import Image
import torchvision.transforms as transform
from torch.utils.data import Dataset
import numpy as np
import torch

class highres_img_dataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.lower().endswith('.jpg')]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        hr_image = Image.open(img_path)
        
        # For some reason when images are displayed in YCbCr only the Cr channel is displayed and the model trains on it
        # I need to research this more and figure out why this is happening
        # Due to this, I will just use RGB for now and accept color variations in the output from the model
        
        # hr_image = hr_image.convert('YCbCr')
        hr_image = Image.open(img_path).convert('RGB')
        
        lr = transform.Compose([
            transform.Resize((720, 1280)),
            transform.ToTensor()
        ])

        # Convert to torch tensors
        hr = transform.Compose([
            transform.ToTensor()
        ])
        
        lr_image_tensor = lr(hr_image)       
        hr_image_tensor = hr(hr_image)

        # Output is a tuple the tensors for the low resolution and high resolution images normalized between 0 and 1
        return lr_image_tensor, hr_image_tensor

