import numpy as np

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

        sar_img = torch.from_numpy(np.load(sar_path)).float()
        opt_img = torch.from_numpy(np.load(opt_path)).float()

        if self.transform:
            sar_img, opt_img = self.transform(sar_img, opt_img)

        return sar_img, opt_img