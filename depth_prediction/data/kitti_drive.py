import os

try:
    from .kitti_drive_segment import KittiDriveSegment
except ImportError:
    from kitti_drive_segment import KittiDriveSegment

class KittiDrive:

    def __init__(self, dir_drive):

        self.dir_drive = dir_drive

        self.drive_segments = []

        self.length = -1

        self.get_segments()

    def get_segments(self):

        roots, dirs, files = next(os.walk(self.dir_drive))

        for dir_segment in dirs:
            self.drive_segments.append(KittiDriveSegment(os.path.join(self.dir_drive, dir_segment)))


    def get_number_drive_segments(self):
        return len(self.drive_segments)

    def __len__(self):
        if self.length != -1:
            return self.length

        images_in_segments = 0
        for segment in self.drive_segments:
            images_in_segments += len(segment)

        self.length = images_in_segments

        return images_in_segments

    def __getitem__(self, idx):
        idx_curr = 0
        for segment in self.drive_segments:
            if idx_curr <= idx < idx_curr + len(segment):
                return segment[idx - idx_curr]
            idx_curr += len(segment)

        return None

