import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
import os
import torchvision.transforms as transform



def train_model(model, n_epochs, training_dataloader, test_every=2, from_checkpoint=False, checkpoint_path=None, save=True, version=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    version_str = f'_{version}' if version else ''
    # Initialize model, optimizer, scaler FIRST
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=.001)
    scaler = GradScaler()
    
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
        print(f'Resuming from epoch {start_epoch}')
    
    # Training setup
    torch.cuda.empty_cache()
    sample_lr = next(iter(training_dataloader))[0].to(device)  # Simplified
    
    avg_loss = 0.0
    try:
        for epoch in tqdm(range(start_epoch, n_epochs)):
            model.train()
            training_loss = 0.0
            total_samples = 0
            
            for lr_imgs, hr_imgs in training_dataloader:
                lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                optimizer.zero_grad()
                
                with autocast():
                    outputs = model(lr_imgs)
                    hr_resized = F.interpolate(hr_imgs, size=(1080, 1920), mode='bilinear')
                    loss = criterion(outputs, hr_resized)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                training_loss += loss.item() * lr_imgs.size(0)
                total_samples += lr_imgs.size(0)
            
            # Epoch-end logic
            avg_loss = training_loss / total_samples
            print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
            
            if (epoch) % test_every == 0:
                model.eval()
                with torch.no_grad(), autocast():
                    pred = model(sample_lr)
                img = transform.ToPILImage()(pred[2])
                # img.save(f'images/training_output/epoch_{epoch+1}{version_str}{date}.jpg')
                img.save(f'images/training_output/epoch_{epoch+1}{version_str}.jpg')
                print('Saved training output to images/training_output')
            
            
            # Save periodic checkpoint
            # if save and (epoch + 1) % 5 == 0:
            #     _save_checkpoint(model=model, epoch=epoch, loss=avg_loss, optimizer=optimizer, scaler=scaler, save_path=f'checkpoints/epoch_{epoch+1}{version_str}.pt')
                
    except KeyboardInterrupt:
        print("\nInterrupted! Saving checkpoint...")
        _save_checkpoint(model=model, epoch=epoch, loss=avg_loss, optimizer=optimizer, scaler=scaler, save_path=f'checkpoints/interrupted_checkpoint_epoch_{epoch}{version_str}.pt')
        return model  # Exit early
    
    # Final save
    if save:
        _save_checkpoint(model=model, epoch=epoch, loss=avg_loss, optimizer=optimizer, scaler=scaler, save_path=f'checkpoints/epoch_{epoch+1}{version_str}.pt')
    
    return model


def test_model(model, checkpoint_path, dataloader, batch_index=0, batch_item=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    latest_epoch = checkpoint['epoch']
    

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
        
    output_dir = f'images/comparisons_at_epoch/epoch_{latest_epoch}_batch{batch_index}_idx{batch_item}/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
            
    lr_img.save(f'{output_dir}lowres.jpg')
    hr_img.save(f'{output_dir}highres.jpg')
    upscaled.save(f'{output_dir}upscaled.jpg')
    print(f'Saved images to {output_dir}')
    
    
def _save_checkpoint(model, epoch, loss, optimizer, scaler, save_path):
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss,
        'optimizer_config': {'lr': optimizer.param_groups[0]['lr']} ,
    }, save_path)