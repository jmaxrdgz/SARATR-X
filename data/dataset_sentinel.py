import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from pathlib import Path


class SentinelDataset(Dataset):
    def __init__(self, data_path: str, terrains=None, transform=None):
        """Dataset for Sentinel-1 and Sentinel-2 image pairs.
        Args:
            data_path (str): Path to the root data directory.
            terrains (str, optional): Specific terrain type to filter. If None, use all terrains.
            transform (callable, optional): Transform to apply to the images.
        """
        super().__init__()
        self.data_path = Path(data_path)
        self.terrains = terrains
        self.transform = transform

        # If no specific terrain is provided, list all terrains in the data path
        if not self.terrains:
            self.terrains = [d for d in self.data_path.iterdir() if d.is_dir()]

        self.pairs = ([], []) # (sar_paths, opt_paths)

        for terrain in self.terrains:
            sar = list((terrain / "s1").iterdir())
            opt = list((terrain / "s2").iterdir())

            for s, o in zip(sorted(sar), sorted(opt)):
                self.pairs[0].append(s)
                self.pairs[1].append(o)
    
    def __len__(self):
        return len(self.pairs[0])
    
    def __getitem__(self, idx):
        sar_path = self.pairs[0][idx]
        opt_path = self.pairs[1][idx]
        
        # Load .jpg images using PIL and convert to tensors
        sar_img = Image.open(sar_path)
        opt_img = Image.open(opt_path)

        # Convert to numpy arrays (no normalization)
        sar_img = np.array(sar_img).astype(np.float32)
        opt_img = np.array(opt_img).astype(np.float32)

        # Handle channel dimensions
        # If grayscale (H, W), add channel dimension -> (1, H, W)
        if sar_img.ndim == 2:
            sar_img = sar_img[np.newaxis, :, :]
        else:
            # If RGB (H, W, C), transpose to (C, H, W)
            sar_img = np.transpose(sar_img, (2, 0, 1))

        if opt_img.ndim == 2:
            opt_img = opt_img[np.newaxis, :, :]
        else:
            opt_img = np.transpose(opt_img, (2, 0, 1))
            
        # Convert to torch tensors
        sar_img = torch.from_numpy(sar_img).float()
        opt_img = torch.from_numpy(opt_img).float()

        if self.transform:
            sar_img, opt_img = self.transform(sar_img, opt_img)

        return sar_img, opt_img