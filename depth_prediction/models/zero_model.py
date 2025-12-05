import torch
import torch.nn as nn

class ZeroBaseline(nn.Module):
    def __init__(self):
        super(ZeroBaseline, self).__init__()
        torch.random.seed = 2147483647
    def forward(self, x):
        
        batch_size = x.shape[0]
        # torch.rand returns values from a uniform distribution [0, 1)
        return torch.zeros(batch_size, 1, 96, 320, device=x.device)
    
    