import torch
import torch.nn.functional as F

class FusionBlock(torch.nn.Module):
    def __init__(self, channels = 256):
        super().__init__()

        # We want to merge low-resolution features from
        # deeper layers with higher resolution features from earlier layers
        # The paper uses convolution blocks to do this and upsampling.

        # Standard residual block layout from resnet.
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(channels)


    def forward(self, x_higher_res, x_lower_res):

        # We want to upsample the lower feature matrix to be the same size as the
        # higher resolution feature matrix.
        x_lower_upsampled = F.interpolate(x_lower_res, size = x_higher_res.shape[-2:], mode = "bilinear"
        , align_corners = False)

        # This is a residual connection, we will save the input to the following Residual block.
        out = x_higher_res + x_lower_upsampled

        residual = out

        # This is a standard residual block from resnet.
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual # We add the input back to the output.

        out = self.relu(out)

        return out




