import torch.nn as nn
import torch


class resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()
        
        
        if in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        else:
            self.downsample = None
        
    def forward(self, x):
        identity = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.downsample:
            identity = self.downsample(identity)
            
        x += identity
        x = self.silu(x)
        
        return x
        


class fastUpscaler(nn.Module):
    # final layer must have C X R^2 channels where R is upscale factor
    def __init__(self, in_channels, channels, upscale_factor=3):
        super(fastUpscaler, self).__init__()
        self.in_channels = in_channels
        self.upscale_factor = upscale_factor
        self.hidden_channels = channels // 2
        self.out_channels = int(3 * (upscale_factor ** 2))
        
        self.prelim = nn.Sequential(
            resblock(in_channels, channels),
            resblock(channels, self.hidden_channels)
        )
        
        self.pixel_shuffle = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor)
        )
        
    def forward(self, x):
        x = self.prelim(x)
        x = self.pixel_shuffle(x)
        
        x = torch.clamp(x, 0.0, 1.0)
        
        return x
    
def test():
    x = torch.randn((3, 3, 640, 360))
    model = fastUpscaler(in_channels=3, channels=64, upscale_factor=3)
    
    preds = model(x)
    
    print(preds.shape)
    print(x.shape)
    
if __name__ == '__main__':
    test()
        
        
        
        
        