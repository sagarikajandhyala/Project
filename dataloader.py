# training/dataloader.py
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class PixelDataset(Dataset):
    def __init__(self, image_dir, patch_size=16):
        self.image_dir = image_dir
        self.files = sorted(os.listdir(image_dir))
        self.patch_size = patch_size
        self.radius = patch_size // 2

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = cv2.imread(
            os.path.join(self.image_dir, self.files[idx]),
            cv2.IMREAD_GRAYSCALE
        ).astype(np.float32) / 255.0

        h, w = img.shape
        i = np.random.randint(self.radius, h - self.radius)
        j = np.random.randint(self.radius, w - self.radius)

        patch = img[i-self.radius:i+self.radius,
                    j-self.radius:j+self.radius]

        target = img[i, j]

        patch = torch.tensor(patch).unsqueeze(0)
        target = torch.tensor(target).unsqueeze(0)

        return patch, target


def get_loader(path, batch_size=16):
    dataset = PixelDataset(path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
