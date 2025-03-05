import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch
import cv2
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class fastUpscaler_dataset(Dataset):
    def __init__(self, image_dir, crop_size=(720, 1280), upscale_factors=[3, 4, 6], transform=None):
        super(fastUpscaler_dataset, self).__init__()
        self.image_dir = image_dir
        self.transform = transform
        self.images = [os.path.join(image_dir, file) for file in os.listdir(self.image_dir)]
        self.upscale_factors = upscale_factors
        self.random_crop = A.RandomCrop(height=crop_size[0], width=crop_size[1])
        self.to_tensor = A.Compose([
            A.Normalize(
                mean=[0, 0, 0],
                std=[1, 1, 1]
            ),
            A.ToTensorV2()
        ])
        
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_file = self.images[idx]
        hr = np.array(Image.open(image_file).convert('RGB'))
        cropped_hr = self.random_crop['image']
        
        if self.transform is not None:
            cropped_hr = self.transform(image=cropped_hr)['image']
            
        downscale_factor = np.random.choice(self.upscale_factors)   
        hr_height, hr_width = cropped_hr.shape[:2]
        lr_height, lr_width = hr_height // downscale_factor, hr_width // downscale_factor
        
        lr_transform = A.Resize(height=lr_height, width=lr_width, interpolation=cv2.INTER_CUBIC)
        lr_image = lr_transform(image=cropped_hr)['image']
        
        hr_tensor = self.to_tensor(image=cropped_hr)['image']
        lr_tensor = self.to_tensor(image=lr_image)['image']
        
        return lr_tensor, hr_tensor
        
        
        
        
            
            
            
random_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.OneOf([
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        A.Blur(blur_limit=3, p=0.5)
    ])
])

