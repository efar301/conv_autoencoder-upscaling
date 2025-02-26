import torch
import torch.nn as nn
import torch.nn.functional as F

class berhu_loss_func(nn.Module):
    def __init__(self, threshold=0.2):
        super(berhu_loss_func, self).__init__()
        self.threshold = threshold

    def forward(self, input, target):

        # Compute the absolute difference
        diff = torch.abs(input - target)

        c = self.threshold * diff.max().item()

        # Apply berhu loss calculation
        loss = torch.where(diff <= c, diff, (diff**2 + c**2) / (2 * c))
        return loss.mean()