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
    
    # Returns RGB image tensors
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        hr_image = Image.open(img_path)
        
        # For some reason when images are displayed in YCbCr 
        # I need to research this more and figure out why this is happening
        # Due to this, I will just use RGB for now and accept color variations in the output from the model
        
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
        
        assert torch.min(lr_image_tensor) >= 0.0 and torch.max(lr_image_tensor) <= 1.0, "LR image tensor not in range [0, 1]"
        assert torch.min(hr_image_tensor) >= 0.0 and torch.max(hr_image_tensor) <= 1.0, "HR image tensor not in range [0, 1]"

        # Output is a tuple the tensors for the low resolution and high resolution images normalized between 0 and 1
        return lr_image_tensor, hr_image_tensor

    # Returns YCbCr image tensors
    
    # def __getitem__(self, idx):
    #     img_path = self.image_files[idx]
        
    #     # Open and convert the image to YCbCr
    #     hr_image = Image.open(img_path).convert('YCbCr')
        
    #     # Create a low resolution version using bicubic interpolation
    #     lr_image = hr_image.resize((1280, 720), Image.BICUBIC)
        
    #     # Convert images to numpy arrays and normalize to [0, 1]
    #     hr_np = np.array(hr_image).astype(np.float32) / 255.0
    #     lr_np = np.array(lr_image).astype(np.float32) / 255.0
        
    #     # Convert numpy arrays to torch tensors with channel-first format
    #     hr_tensor = torch.from_numpy(hr_np).permute(2, 0, 1)
    #     lr_tensor = torch.from_numpy(lr_np).permute(2, 0, 1)
        
    #     return lr_tensor, hr_tensor