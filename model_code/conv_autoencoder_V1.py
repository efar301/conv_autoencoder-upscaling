import torch.nn as nn
import torch.nn.functional as F

# Formulas for determining output size of convolutional layers assuming square kernel, stride, and padding

# DOWNSCALING
# new_rows = floor((rows - kernel_size + 2 * padding) / stride) + 1
# new_cols = floor((cols - kernel_size + 2 * padding) / stride) + 1

# UPSCALING
# new_rows = ((rows - 1) * stride + kernel_size - 2 * padding)
# new_cols = ((cols - 1) * stride + kernel_size - 2 * padding)

class conv_autoencoder_V1(nn.Module):
    def __init__(self, version=''):
        super(conv_autoencoder_V1, self).__init__()
        self.version = version
        
        # Encoder
        # Input is 1280 x 720 with 3 channels
        self.encoder = nn.Sequential(
            # Downscale by stride amount
            # In this case it goes to 640 x 360 with 32 channels
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Mish(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Mish(inplace=True),
            
            nn.Conv2d(64, 90, kernel_size=3, stride=1, padding=1),
            nn.Mish(inplace=True)
        )
        # Decoder
        # Needs to upscale to 1920 x 1080 with 3 channels
        self.decoder = nn.Sequential(
            # Upscale by stride amount 
            nn.ConvTranspose2d(
                in_channels=90,
                out_channels=64,
                kernel_size=5,      # Kernel size 5 ensures exact 3x scaling
                stride=3,           # Upsamples by 3x
                padding=1,          # Compensates for kernel size
            ),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # Anti-aliasing to smooth edges
            nn.Mish(inplace=True),
            
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Mish(inplace=True),
            
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            
            # Sigmoid because output is between 0 and 1
            nn.Sigmoid()
        )
        
    def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

