import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = timm.create_model("vit_base_patch16_384", pretrained=True, dynamic_img_size=True)

        # We will save the data from these intermediate layers to pass to the decoder.
        # This base model is composed of 12 transformer layers.
        self.intermediate_layers = [2, 5, 8, 11]

        self.embed_dim = self.model.embed_dim

    def forward(self, x):

        # We will convert our input image into a series of 16x16 patches (this is resolution invariant)
        # as long as the resolution is divisible by 16.
        x = self.model.patch_embed(x)

        # We add the [CLS] token, this is to maintain compatibility with the initial
        # model being trained on ImageNet for classification and to preserve the
        # positional embedding learned.


        # We use the model's internal _pos_embed method.
        # This is CRITICAL because it handles the interpolation of the position
        # embeddings for the KITTI resolution (1216x352) automatically.

        x = self.model._pos_embed(x)
        # Now we add the positional embedding to the input tokens.
        # We will randomly zero out some of the tokens being inputted into the transformer
        # Pass through blocks and extract specific layers

        features = []
        for i, block in enumerate(self.model.blocks):
            x = block(x)
            if i in self.intermediate_layers:
                features.append(x)

        # We will return these features. The output from the other blocks is not used outside this model.
        return features