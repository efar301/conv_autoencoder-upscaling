import torch.nn as nn

class conv_autoencoder(nn.Module):
    def __init__(self):
        super(conv_autoencoder, self).__init__()
        
        # encoder
        # input is 3 x 1280 x 720
        self.encoder = nn.Sequential(
            # downsize by 2x to 64 x 640 x 360 
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.Mish(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Mish(inplace=True)
        )
        # decoder
        # needs to upscale to 1920 x 1080
        self.decoder = nn.Sequential(
            # upscale 3x 
            nn.Upsample(scale_factor=3, mode='bilinear'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.Mish(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

