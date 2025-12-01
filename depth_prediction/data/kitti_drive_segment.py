import os

from torchvision import transforms
from PIL import Image
import numpy as np

class KittiDriveSegment:

    def __init__(self, dir_segment, target_img_size = (352, 1216)):

        self.dir_segment = dir_segment

        self.image_rgb_dir = os.path.join(self.dir_segment, "image_02/data")

        self.image_depth_dir = os.path.join(self.dir_segment, "proj_depth/groundtruth/image_02")

        self.images_rgb = sorted(os.listdir(self.image_rgb_dir))

        self.images_depth = sorted(os.listdir(self.image_depth_dir))

        assert len(self.images_rgb) == len(self.images_depth)

        self.target_img_size = target_img_size

        self.rgb_transform = transforms.Compose([
            transforms.Resize(self.target_img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # This comes from imagenet
        ])

        self.depth_transform = transforms.Compose([
            transforms.Resize(self.target_img_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images_rgb)

    def __getitem__(self, idx):

        # We load the image on demand
        image_rgb_path = os.path.join(self.image_rgb_dir, self.images_rgb[idx])
        image_depth_path = os.path.join(self.image_depth_dir, self.images_depth[idx])

        image_rgb = Image.open(image_rgb_path).convert("RGB")
        image_depth = Image.open(image_depth_path)

        image_tensor_rgb = self.rgb_transform(image_rgb)
        image_tensor_depth = self.depth_transform(image_depth)

        image_tensor_depth = image_tensor_depth / 256.0

        return image_tensor_rgb, image_tensor_depth








