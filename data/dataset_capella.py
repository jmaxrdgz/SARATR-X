from pathlib import Path
import numpy as np

import torch
from torch.utils.data import Dataset


class CapellaDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.files = sorted(Path(folder).glob("*.npy"))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr = np.load(self.files[idx])
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]

        img = torch.from_numpy(arr).float()

        if self.transform:
            img = self.transform(img)

        assert img.shape[0] in [1, 3], "Image must have 1 or 3 channels"
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)

        return img, 0