import torch
import torch.nn as nn

class ReassembleBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()

        self.scale_factor = scale_factor

        # We will use this block to transform a 1D sequence of tokens outputted by
        # the transformer to a 2d Map of features.

        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # We want to either upsample or downsample the output of the transformer depending
        # on which transformer layer we are transforming data from.
        if scale_factor != 1.0:
            self.resize = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=False)
        else:
            self.resize = nn.Identity()

        # Paper refines the output afterwards with a convolution
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, H, W, patch_size=16):

        # x is of dimensions [# Batches, Num_tokens, Size of 1-d vector]

        # We want to remove the CLS token from this sequence because it is not useful for us.
        x = x[:, 1:, :]

        # We want to calculate how many patches fit into the original image
        h_patches = H // patch_size
        w_patches = W // patch_size

        B, N, C = x.shape

        # Now we reshape the sequence of tokens into a 2D matrix of patches.
        x = x.transpose(1,2).reshape(B, C, h_patches, w_patches)

        # Resize the patches to output channel size
        x = self.project(x)
        x = self.resize(x)
        x = self.conv(x)

        return x
        


