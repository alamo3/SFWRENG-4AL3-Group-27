import torch
import torch.nn as nn
import torchvision.models as models

class DepthNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder: ResNet-34
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # Keep only the convolutional layers
        self.conv1 = resnet.conv1   # (64 channels, /2)
        self.bn1   = resnet.bn1
        self.relu  = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1   # 64 channels  (/4)
        self.layer2 = resnet.layer2   # 128 channels (/8)
        self.layer3 = resnet.layer3   # 256 channels (/16)
        self.layer4 = resnet.layer4   # 512 channels (/32)

        # Decoder
        self.up4 = self._upsample_block(512, 256)  # /16
        self.up3 = self._upsample_block(256, 128)  # /8
        self.up2 = self._upsample_block(128, 64)   # /4
        self.up1 = self._upsample_block(64, 64)    # /2

        # Final depth prediction layer
        self.final = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def _upsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # ---- Encoder ----
        x0 = self.relu(self.bn1(self.conv1(x)))  # /2
        x1 = self.layer1(self.maxpool(x0))       # /4
        x2 = self.layer2(x1)                     # /8
        x3 = self.layer3(x2)                     # /16
        x4 = self.layer4(x3)                     # /32

        # Decoder
        d4 = self.up4(x4) + x3
        d3 = self.up3(d4) + x2
        d2 = self.up2(d3) + x1
        d1 = self.up1(d2) + x0

        # Depth output — ensure non-negative depth
        depth = torch.relu(self.final(d1))
        return depth
