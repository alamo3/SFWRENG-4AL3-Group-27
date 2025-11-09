import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class KITTIDepthSelectionDataset(Dataset):
    def __init__(self, root_dir):
        self.image_dir = os.path.join(root_dir, "image")
        self.depth_dir = os.path.join(root_dir, "depth")

        self.images = sorted(os.listdir(self.image_dir))
        self.depths = sorted(os.listdir(self.depth_dir))

        self.img_transform = transforms.Compose([
            transforms.Resize((192, 640)),
            transforms.ToTensor(),
            transforms.Normalize([0.45,0.45,0.45],[0.225,0.225,0.225]),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # ---- Load RGB ----
        img = Image.open(os.path.join(self.image_dir, self.images[idx])).convert("RGB")
        img = self.img_transform(img)

        # ---- Load Depth (16-bit PNG) ----
        depth = Image.open(os.path.join(self.depth_dir, self.depths[idx]))

        # Convert to float meters *before resize*
        depth = np.array(depth).astype(np.float32) / 256.0

        # Convert back to PIL for resizing
        depth = Image.fromarray(depth)

        # Resize *with nearest neighbor to avoid interpolation artifacts*
        depth = depth.resize((320, 96), Image.NEAREST)

        # Convert to Tensor
        depth = torch.from_numpy(np.array(depth)).unsqueeze(0)

        return img, depth
