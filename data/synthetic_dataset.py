import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class SyntheticRectDataset(Dataset):
    def __init__(self, n=2000, img_size=224):
        self.n = n
        self.img_size = img_size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        x1, y1 = np.random.randint(20, 120, size=2)
        w, h = np.random.randint(30, 80, size=2)
        x2, y2 = x1 + w, y1 + h

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)

        # Normalize bbox
        cx = (x1 + x2) / 2 / self.img_size
        cy = (y1 + y2) / 2 / self.img_size
        bw = w / self.img_size
        bh = h / self.img_size

        target = torch.tensor([cx, cy, bw, bh, 1.0, 1.0])

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img, target
