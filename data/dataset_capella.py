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

        # Return a dummy tensor placeholder instead of scalar 0
        # This maintains shape compatibility even though it won't be used in MGF mode
        dummy_target = torch.zeros_like(img)
        return img, dummy_target