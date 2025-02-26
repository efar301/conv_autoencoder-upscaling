import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from PIL import Image
import os
import torchvision.transforms as transform
from pytorch_msssim import SSIM
import torch.nn.init as init
import torch.nn.functional as F
from piq import SSIMLoss
from model_code.berhu_loss import berhu_loss_func
from datetime import datetime


def train_model(model, n_epochs, training_dataloader, learning_rate=.001, test_every=2, upscale_amt=1.5, from_checkpoint=False, checkpoint_path=None, save=True, version=None, version_change=''):
    '''
    returns total_losses, berhu_epoch_loss
    '''
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    version_str = f'_{version}_{version_change}' if version else ''
    # Initialize model, optimizer, scaler
    model.to(device)
    berhu_loss = berhu_loss_func()
    ssim_loss = SSIMLoss(data_range=1.0, reduction='mean').to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()
    total_losses = []
    
    # Load checkpoint if needed
    start_epoch = 0
    if from_checkpoint:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f'Checkpoint {checkpoint_path} not found')
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        version = checkpoint['version']
        version_change = checkpoint['version_change']
        total_losses = checkpoint['total_losses']
        print(f'Resuming from epoch {start_epoch}')
    
    # Training setup
    torch.cuda.empty_cache()
    sample_lr = next(iter(training_dataloader))[0].to(device)  # Simplified
    
    avg_loss = 0.0
    berhu_losses = []
    low_res_losses = []
    high_res_losses = []
    if total_losses is None:
        total_losses = []   
    try:
        for epoch in tqdm(range(start_epoch, n_epochs)):
            model.train()
            training_loss = 0.0
            total_samples = 0
            
            for lr_imgs, hr_imgs in training_dataloader:
                lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                optimizer.zero_grad()
                
                # FP16 training since its faster
                with autocast():
                    outputs_1080p = model(lr_imgs, target_resolution='1080p')
                    assert torch.min(outputs_1080p) >= 0.0 and torch.max(outputs_1080p) <= 1.0, 'Model outputs out of range'
                    hr_resized = F.interpolate(hr_imgs, size=(outputs_1080p.shape[2], outputs_1080p.shape[3]), mode='bilinear')
                    
                    
                    outputs_1440p = model(lr_imgs, target_resolution='1440p')
                    assert torch.min(outputs_1440p) >= 0.0 and torch.max(outputs_1440p) <= 1.0, 'Model outputs out of range'

                    berhu_epoch_loss_1080p = berhu_loss(outputs_1080p, hr_resized)
                    berhu_epoch_loss_1440p = berhu_loss(outputs_1440p, hr_imgs)
                    
                    low_res_losses.append(berhu_epoch_loss_1080p.item())
                    high_res_losses.append(berhu_epoch_loss_1440p.item())
                    
                    
                    loss = .5 * berhu_epoch_loss_1080p + .5 * berhu_epoch_loss_1440p
                    total_losses.append(loss.item())
                    
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                training_loss += loss.item() * lr_imgs.size(0)
                total_samples += lr_imgs.size(0)
            
            # Epoch end logic
            avg_loss = training_loss / total_samples
            print(f'Epoch {epoch+1}, Loss: {avg_loss:.7}')
            
            # Save progress image
            save_rgb(epoch, model, sample_lr, version, version_change, test_every)
            log_progress(epoch, berhu_epoch_loss_1080p, berhu_epoch_loss_1440p, avg_loss, learning_rate, version, version_change)
            
            # Checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                _save_checkpoint(model=model, epoch=epoch, loss=avg_loss, optimizer=optimizer, scaler=scaler, version=version, version_change=version_change, total_losses=total_losses, save_path=f'checkpoints/{version}_{version_change}/epoch_{epoch+1}{version_str}.pt')  
                
    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint...")
        _save_checkpoint(model=model, epoch=epoch, loss=avg_loss, optimizer=optimizer, scaler=scaler, version=version, version_change=version_change, total_losses=total_losses, save_path=f'checkpoints/{version}_{version_change}/epoch_{epoch+1}{version_str}.pt')
        torch.cuda.empty_cache()
        return total_losses, berhu_epoch_loss_1080p, berhu_epoch_loss_1440p  # Exit early
    
    except Exception as e:  # Save checkpoint on error
        print(f'Error: {e}')
        _save_checkpoint(model=model, epoch=epoch, loss=avg_loss, optimizer=optimizer, scaler=scaler, version=version, version_change=version_change, total_losses=total_losses, save_path=f'checkpoints/{version}_{version_change}/epoch_{epoch+1}{version_str}.pt')
        torch.cuda.empty_cache()
        return total_losses, berhu_epoch_loss_1080p, berhu_epoch_loss_1440p
    
    # Final save
    if save:
        _save_checkpoint(model=model, epoch=epoch, loss=avg_loss, optimizer=optimizer, scaler=scaler, version=version, version_change=version_change, total_losses=total_losses, save_path=f'checkpoints/{version}_{version_change}/epoch_{epoch+1}{version_str}.pt')
    
    torch.cuda.empty_cache()
    return total_losses, berhu_epoch_loss_1080p, berhu_epoch_loss_1440p


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)

def log_progress(epoch, low_res_loss, high_res_loss, total_loss, learning_rate, version, version_change):    
    progress_dir = f'training_info/training_output/{version}_{version_change}'
    if not os.path.exists(progress_dir):
        os.makedirs(progress_dir, exist_ok=True)
    
    file_exists = os.path.exists(f'{progress_dir}/{version}_{version_change}.log')  
    date = datetime.now().strftime('%m-%d-%Y_%H:%M:%S')
    
    with open(f'{progress_dir}/{version}_{version_change}.log', 'a') as f:
        
        if not file_exists:
            f.write('Epoch,Low_Res_Loss,High_Res_Loss,Total_Loss,Learning_Rate,Time(mm-dd-yyyy_hh:mm:ss)\n')
            
        f.write(f'{epoch},{low_res_loss:.7f},{high_res_loss:.7f},{total_loss:.7f},{learning_rate},{date}\n')   
            

def test_model(model, checkpoint_path, dataloader, batch_index=0, batch_item=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    latest_epoch = checkpoint['epoch']
    version = checkpoint['version']
    version_changes = checkpoint['version_change']  

    sample_batch = None
    for i, batch in enumerate(dataloader):
        if i == batch_index:
            sample_batch = batch
            break
    
    sample_lr, sample_hr = sample_batch
    sample_lr = sample_lr.to(device)
    sample_hr = sample_hr.to(device)
            
    model.eval()
    model.to(device)   
    with torch.no_grad():
        # Generate prediction
        with autocast():
            pred = model(sample_lr)
        
        to_img = transform.ToPILImage()
        lr_img = to_img(sample_lr[batch_item])
        upscaled = to_img(pred[batch_item])
        hr_img = to_img(sample_hr[batch_item])
        
    output_dir = f'training_info/comparisons_at_epoch/epoch_{latest_epoch}_batch{batch_index}_idx{batch_item}_{version}_{version_changes}/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
            
    lr_img.save(f'{output_dir}lowres.jpg')
    hr_img.save(f'{output_dir}highres.jpg')
    upscaled.save(f'{output_dir}upscaled.jpg')
    print(f'Saved images to {output_dir}')
    
    
def _save_checkpoint(model, epoch, loss, optimizer, scaler, total_losses, save_path, version, version_change):
    if not os.path.exists(os.path.dirname(save_path)):  # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True) 
    
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss,
        'optimizer_config': {'lr': optimizer.param_groups[0]['lr']} ,
        'version': version,
        'version_change': version_change,
        'total_losses': total_losses
    }, save_path)
        
def save_rgb(epoch, model, sample_lr, version, version_change, test_every):
        progress_dir = f'training_info/training_output/{version}_{version_change}/progress_images'
        if not os.path.exists(progress_dir):
            os.makedirs(progress_dir, exist_ok=True)
            
        dir_1080p = f'{progress_dir}/1080p'
        dir_1440p = f'{progress_dir}/1440p'
        
        if not os.path.exists(dir_1080p):
            os.makedirs(dir_1080p, exist_ok=True)   
        if not os.path.exists(dir_1440p):
            os.makedirs(dir_1440p, exist_ok=True)   
                
        if (epoch) % test_every == 0:
            model.eval()
            with torch.no_grad(), autocast():
                pred_1080p = model(sample_lr, target_resolution='1080p')
                pred_1440p = model(sample_lr, target_resolution='1440p')    
            # Convert the image from tensor to PIL image
            img_1080p = transform.ToPILImage()(pred_1080p[2])
            img_1440p = transform.ToPILImage()(pred_1440p[2])
            
            # Save image
            img_1080p.save(f'{dir_1080p}/epoch_{epoch+1}_{version}_1080p.jpg')
            img_1440p.save(f'{dir_1440p}/epoch_{epoch+1}_{version}_1440p.jpg')
            print(f'Saved training output to {progress_dir}')