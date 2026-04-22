import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class KITTIDataset(Dataset):
    def __init__(self, img_dir, lidar_dir):
        self.img_dir = img_dir
        self.lidar_dir = lidar_dir
        self.files = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.files)

    def lidar_to_depth(self, lidar_path, img_shape=(640, 640)):
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        depth = np.zeros(img_shape)

        for p in points:
            x, y, z, _ = p
            if z > 0:
                u = int(x * 10) % img_shape[1]
                v = int(y * 10) % img_shape[0]
                depth[v, u] = z

        return depth

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.files[idx])
        lidar_path = os.path.join(self.lidar_dir, self.files[idx].replace(".png", ".bin"))

        img = cv2.imread(img_path)
        img = cv2.resize(img, (640, 640))
        img = img / 255.0

        depth = self.lidar_to_depth(lidar_path)
        depth = np.expand_dims(depth, axis=2)

        img = np.transpose(img, (2, 0, 1))
        depth = np.transpose(depth, (2, 0, 1))

        return torch.tensor(img, dtype=torch.float32), torch.tensor(depth, dtype=torch.float32)