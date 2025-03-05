import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

from fastUpscaler import fastUpscaler
from fastUpscaler_dataset import fastUpscaler_dataset, random_transform 

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = fastUpscaler_dataset(
        image_dir=args.image_dir,
        transform=random_transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    model = fastUpscaler(
        in_channels=3,
        channels=args.channels
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    writer = SummaryWriter(log_dir=args.log_dir)
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    scaler = torch.cuda.amp.GradScaler()
    
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        loop = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch + 1}/{args.epochs}', ncols=200)
        for batch_idx, (lr, hr, scale_factor) in loop:
            scale_factor = scale_factor[1].item()
            lr, hr = lr.to(device), hr.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                preds = model(lr, scale_factor)
                loss = criterion(preds, hr)
                
                
            scaler.scale(loss).backward()
            scaler.step(optimizer) 
            scaler.update()
            
            running_loss += loss.item()
            global_step += 1
            
            if batch_idx % args.log_interval == 0:
                avg_loss = running_loss / (batch_idx + 1)
                writer.add_scalar('Training Loss', avg_loss, global_step)
                grid = vutils.make_grid(preds, normalize=True, scale_each=True)
                writer.add_image('Predictions', grid, global_step)
                loop.set_postfix({'Loss': f'{avg_loss:.5f}'})
                
         
        if (epoch + 1) % 10 == 0:      
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss / len(dataloader)
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
            
    writer.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train FastUpscaler model")
    parser.add_argument('--image_dir', type=str, default='train_test_data/training_set', help='Directory containing 4K images')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for TensorBoard logs')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers')
    parser.add_argument('--channels', type=int, default=64, help='Number of channels for the model')
    parser.add_argument('--crop_height', type=int, default=720, help='Height of HR crop')
    parser.add_argument('--crop_width', type=int, default=1280, help='Width of HR crop')
    parser.add_argument('--log_interval', type=int, default=10, help='Interval (in batches) to log training loss')

    args = parser.parse_args()
    train(args)
  