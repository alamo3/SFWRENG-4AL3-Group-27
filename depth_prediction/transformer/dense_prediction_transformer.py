import torch
import torch.nn as nn
import torch.nn.functional as F
from .vision_transformer import VisionTransformer

from .reassemble_block import ReassembleBlock
from .fusion_block import FusionBlock

class DensePredictionTransformer(nn.Module):

    def __init__(self, transformer_features):

        super().__init__()
        self.encoder = VisionTransformer()

        self.transformer_features = transformer_features

        self.base_dim = self.encoder.embed_dim

        self.reassemble_1 = ReassembleBlock(self.base_dim, self.transformer_features, scale_factor=4.0)
        self.reassemble_2 = ReassembleBlock(self.base_dim, self.transformer_features, scale_factor=2.0)
        self.reassemble_3 = ReassembleBlock(self.base_dim, self.transformer_features, scale_factor=1.0)
        self.reassemble_4 = ReassembleBlock(self.base_dim, self.transformer_features, scale_factor=0.5)

        self.fusion_3 = FusionBlock(self.transformer_features)
        self.fusion_2 = FusionBlock(self.transformer_features)
        self.fusion_1 = FusionBlock(self.transformer_features)

        self.depth_prediction_head = nn.Sequential(
            nn.Conv2d(self.transformer_features, self.transformer_features // 2, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(self.transformer_features // 2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Delete the unused classification layers to satisfy DDP
        if hasattr(self.model, 'head'):
            del self.model.head  # The classification head
        if hasattr(self.model, 'norm'):
            del self.model.norm  # The final normalization layer
        if hasattr(self.model, 'fc_norm'):
            del self.model.fc_norm  # FC Norm


    def forward(self, x):

        B, C, H, W = x.shape

        tokens_list = self.encoder(x)

        f1 = self.reassemble_1(tokens_list[0], H, W)
        f2 = self.reassemble_2(tokens_list[1], H, W)
        f3 = self.reassemble_3(tokens_list[2], H, W)
        f4 = self.reassemble_4(tokens_list[3], H, W)

        out = self.fusion_1(f3, f4)
        out = self.fusion_2(f2, out)
        out = self.fusion_3(f1, out)

        depth = self.depth_prediction_head(out)

        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=False)

        return depth