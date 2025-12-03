import os
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class KITTIDepthSelectionDataset(Dataset):
    def __init__(self, root_dir):
        """
        root_dir should point at the KITTI split (e.g., depth_prediction/KITTI/KITTI/train).
        Expects images under **/image_0*/data/*.png and depth maps stored alongside each drive
        in one of these common KITTI layouts:
          - <drive>/proj_depth/groundtruth/image_0X/<file>.png
          - <drive>/groundtruth_depth/image_0X/data/<file>.png
          - <drive>/depth/image_0X/data/<file>.png
        """
        self.root_dir = Path(root_dir)
        self.img_transform = transforms.Compose([
            transforms.Resize((192, 640)),
            transforms.ToTensor(),
            transforms.Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
        ])

        self.pairs = []
        for img_path in sorted(self.root_dir.glob("**/image_0[23]/data/*.png")):
            depth_path = self._find_depth_for_img(img_path)
            if depth_path is None:
                continue
            self.pairs.append((img_path, depth_path))

        if not self.pairs:
            raise RuntimeError(f"No image/depth pairs found under {self.root_dir}")

    def _find_depth_for_img(self, img_path: Path) -> Path | None:
        # img_path looks like .../<drive>/image_0X/data/<frame>.png
        drive_dir = img_path.parents[2]
        cam = img_path.parent.parent.name  # image_02 or image_03
        fname = img_path.name

        candidates = [
            drive_dir / "proj_depth" / "groundtruth" / cam / fname,
            drive_dir / "groundtruth_depth" / cam / "data" / fname,
            drive_dir / "depth" / cam / "data" / fname,
        ]
        for cand in candidates:
            if cand.exists():
                return cand
        return None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, depth_path = self.pairs[idx]

        img = Image.open(img_path).convert("RGB")
        img = self.img_transform(img)

        depth = Image.open(depth_path)
        depth = np.array(depth).astype(np.float32) / 256.0
        depth = Image.fromarray(depth)
        depth = depth.resize((320, 96), Image.NEAREST)
        depth = torch.from_numpy(np.array(depth)).unsqueeze(0)

        return img, depth
