import os

import torch
import matplotlib.pyplot as plt

from PIL import Image
try:
    from .kitti_drive import KittiDrive
except ImportError:
    from kitti_drive import KittiDrive


class KITTIDepthLoader:

    def __init__(self, base_dir):
        self.base_dir = base_dir

        self.list_drives = []

        self.get_list_of_drives()

        self.length = -1


    def get_list_of_drives(self):

        roots, dirs, files = next(os.walk(self.base_dir))

        for dir_found in dirs:
            self.list_drives.append(KittiDrive(os.path.join(self.base_dir, dir_found)))


    def __len__(self):

        if self.length != -1:
            return self.length

        num_images = 0
        for drive in self.list_drives:
            num_images += len(drive)

        self.length = num_images

        return num_images

    def __getitem__(self, idx):

        idx_curr = 0
        for drive in self.list_drives:

            if idx_curr <= idx < idx_curr + len(drive):
                return drive[idx - idx_curr]
            idx_curr += len(drive)

        return None


# Test it out

if __name__ == "__main__":
    data_loader = KITTIDepthLoader("D:\\KITTI\\train")
    print(len(data_loader))

    img, label = data_loader[1000]

    f, axarr = plt.subplots(2)
    axarr[0].imshow(img.permute(1, 2, 0))
    axarr[1].imshow(label.squeeze(), cmap="gray")
    plt.show()