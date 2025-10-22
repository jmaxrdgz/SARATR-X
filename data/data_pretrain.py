from pathlib import Path
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import config


class NormalizeSAR:
    """SAR Normalization used in TRANSAR paper."""
    def __init__(self, std_dev):
        self.std_dev = std_dev

    def __call__(self, img):
        img = img - img.mean()
        img = img / self.std_dev
        return img
    

def build_loader():
    transform = transforms.Compose([
        transforms.RandomResizedCrop(config.data.img_size, scale=(0.2, 1.0), interpolation=3),
        transforms.Resize((config.data.img_size, config.data.img_size)),

        # transforms.RandomAffine(degrees=(-10,10), shear=(0, 0, -10, 10)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(contrast=0.5),

        # NormalizeSAR(std_dev=config.data.dataset_std_dev),
    ])

    train_set = NpyDataset(folder=config.data.train_data, transform=transform)
    train_loader = DataLoader(train_set, batch_size=config.train.batch_size, shuffle=True, 
        persistent_workers=True, pin_memory=True, num_workers=config.data.num_workers, drop_last=True)

    return train_loader


class NpyDataset(Dataset):
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