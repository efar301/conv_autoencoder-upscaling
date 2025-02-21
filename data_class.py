import os
from PIL import Image
import torchvision.transforms as transform
from torch.utils.data import Dataset


# change the dimentions to 720p and then to torch tensors
lr = transform.Compose([
    transform.Resize((720, 1280)),
    transform.ToTensor()
])

# convert to torch tensors
hr = transform.Compose([
    transform.ToTensor()
])


class highres_img_dataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.lower().endswith('.jpg')]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        hr_image = Image.open(img_path).convert('YCbCr')
        
        lr = transform.Compose([
            transform.Resize((720, 1280)),
            transform.ToTensor()
        ])

        # convert to torch tensors
        hr = transform.Compose([
            transform.ToTensor()
        ])
        
        lr_image = lr(hr_image)
            
        hr_image = lr(hr_image)
            
        return lr_image, hr_image

