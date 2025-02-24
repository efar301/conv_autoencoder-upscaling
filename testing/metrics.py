from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr  
import numpy as np  

def psnr_score(upscaled, original):
    # downscale original image and calculate score
    upscaled = Image.open(upscaled)
    original = Image.open(original) 
    original = original.resize(upscaled.size)
    
    upscaled_np = np.array(upscaled)
    original_np = np.array(original)
    
    psnr_score = psnr(upscaled_np, original_np)
    return psnr_score   

def ssim_score(upscaled, original):
    # downscale original image and calculate score
    upscaled = Image.open(upscaled)
    original = Image.open(original) 
    original = original.resize(upscaled.size)
    
    upscaled_np = np.array(upscaled)
    original_np = np.array(original)
    
    ssim_score = ssim(upscaled_np, original_np, channel_axis=-1)
    return ssim_score   