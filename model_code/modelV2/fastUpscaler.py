import torch.nn as nn
import torch
import torch.nn.init as init


class resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resblock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()
        
        
        if in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        else:
            self.downsample = None
        
    def forward(self, x):
        identity = x
        
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.conv2(x)
        # x = self.bn2(x)
        
        if self.downsample:
            identity = self.downsample(identity)
            
        x += identity
        x = self.silu(x)
        
        return x
        


class fastUpscaler(nn.Module):
    # final layer must have C X R^2 channels where R is upscale factor
    def __init__(self, in_channels, channels, upscale_factors=[3, 4, 6]):
        super(fastUpscaler, self).__init__()
        self.in_channels = in_channels
        self.upscale_factor = upscale_factors
        self.hidden_channels = channels // 2
        
        self.prelim = nn.Sequential(
            resblock(in_channels, channels),
            resblock(channels, self.hidden_channels)
        )
        
        
        self.heads = nn.ModuleDict()
        for factor in upscale_factors:
                out_channels = 3 * (factor ** 2)
                self.heads[str(factor)] = nn.Sequential(
                    nn.Conv2d(self.hidden_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.PixelShuffle(factor)
                )
                
                icnr_init(self.heads[str(factor)][0].weight, factor)
        
        
    def forward(self, x, scale_factor):
        x = self.prelim(x)
        x = self.heads[str(scale_factor)](x)
        
        x = torch.clamp(x, 0.0, 1.0)
        
        return x
    
    

def icnr_init(tensor, scale=2, init_method=init.kaiming_normal_):

    subkernel_channels = tensor.shape[0] // (scale ** 2)
    subkernel = torch.zeros((subkernel_channels, *tensor.shape[1:]), device=tensor.device)
    init_method(subkernel)
    subkernel = subkernel.repeat(scale ** 2, 1, 1, 1)
    tensor.data.copy_(subkernel)
   
def test():
    x = torch.randn((3, 3, 640, 360))
    model = fastUpscaler(in_channels=3, channels=64, upscale_factors=[3, 4, 6])
    
    preds = model(x, 3)
    
    print(preds.shape)
    print(x.shape)
    
if __name__ == '__main__':
    test()
        
        
        
        
        