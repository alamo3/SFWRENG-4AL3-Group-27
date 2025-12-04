import torch
import torch.nn as nn
import torchvision.models as models
class MLP_perception(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # input RGB
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            #nn.MaxPool2d(2),  # downsample
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # MLP per-pixel
        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # depth output
        )

    def forward(self, x):
        # x shape: (B, 3, H, W)
        features = self.cnn(x)  # (B, C, H', W')
        B, C, H, W = features.shape
        # flatten per pixel
        features = features.permute(0,2,3,1).reshape(-1, C)  # (B*H'*W', C)
        depth = self.mlp(features)  # (B*H'*W', 1)
        depth = depth.view(B,1, H, W)  # (B, H, W)
        return depth