import torch.nn as nn
import torch.nn.functional as F

# Formulas for determining output size of convolutional layers assuming square kernel, stride, and padding

# DOWNSCALING
# new_rows = floor((rows - kernel_size + 2 * padding) / stride) + 1
# new_cols = floor((cols - kernel_size + 2 * padding) / stride) + 1

# UPSCALING
# new_rows = ((rows - 1) * stride + kernel_size - 2 * padding)
# new_cols = ((cols - 1) * stride + kernel_size - 2 * padding)

class multires_conv_autoencoder_V1(nn.Module):
    '''
    Instance of multi_res_conv_autoencoder_V1s
    '''
    def __init__(self, version=''):
        '''
        Defines single encoder layer and multiple decoder layers for model
        
        Params:
            version: version number of model (doesnt do anything right now)
        '''
        super(multires_conv_autoencoder_V1, self).__init__()
        self.version = version
        
        # Encoder
        # Input is 1280 x 720 with 3 channels
        self.encoder = nn.Sequential(
            # Downscale by stride amount
            # In this case it goes to 640 x 360 with 32 channels
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Mish(inplace=True), # Mish activation because it's better than ReLU
            
            # This layer adds more channels
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), 
            nn.Mish(inplace=True),
            
            # This layer adds more channels
            nn.Conv2d(64, 90, kernel_size=3, stride=1, padding=1),
            nn.Mish(inplace=True)
        )
        # Decoder
        # Upscales to 1920 x 1080 with 3 channels
        self.decoder_1080p = nn.Sequential(
            # This layer is an upscaling layer
            nn.ConvTranspose2d(
                in_channels=90,
                out_channels=64,
                kernel_size=5,      # Kernel size 5 ensures exact 3x scaling
                stride=3,           # Upsamples by 3x
                padding=1,          # Compensates for kernel size
            ),
            
            # This layer acts as anti aliasing to smooth checkerboarding that may occur from ConvTranspose2d
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.Mish(inplace=True),
            
            # This layer reduces channels
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Mish(inplace=True),
            
            # This layer returns to final 3 channels
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            
            # Sigmoid because output is between 0 and 1
            nn.Sigmoid()
        )
        
        # Upscales to 2560 x 1440 with 3 channels
        self.decoder_1440p = nn.Sequential(
            # This layer is an upscaling layer 
            nn.ConvTranspose2d(
                in_channels=90,
                out_channels=64,
                kernel_size=4,      # Kernel size 4 ensures exact 4x scaling
                stride=4,           # Upsamples by 4x
                padding=0,          # Compensates for kernel size
            ),
            # This layer acts as anti aliasing to smooth checkerboarding that may occur from ConvTranspose2d
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.Mish(inplace=True),
            
            # This layer reduces channels
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Mish(inplace=True),
            
            # This layer returns to final 3 channels
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            
            # Sigmoid because output is between 0 and 1
            nn.Sigmoid()
        )
        
    def forward(self, x, target_resolution='1080p'):
        '''
        Forward pass through model
        
        Params:
            x: tensor consisting of (batch_size, channels=3, height=720, width=1280), values must be normalized between 0 and 1
            target_resolution: desired output resolution, currently supports 1920x1080 or 2560x1440
            
        Returns:
            x: tensor consisting of (batch_size, chanel, height=target resolution height, width=target resolution width
        '''
        x = self.encoder(x)
            
        # This will never activate but it is a failsafe
        assert target_resolution == '1080p' or target_resolution == '1440p', 'target_resolution must be 1080p or 1440p'
        
        if target_resolution == '1080p':
            x = self.decoder_1080p(x)
        elif target_resolution == '1440p':
            x = self.decoder_1440p(x)
        return x