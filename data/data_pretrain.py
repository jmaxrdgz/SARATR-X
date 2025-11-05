import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import config

from .dataset_capella import CapellaDataset
from .dataset_sentinel import SentinelDataset


# TODO: check normalization values for sentinel-1 and sentinel-2
class NormalizeSAR:
    """SAR Normalization used in TRANSAR paper."""
    def __init__(self, std_dev):
        self.std_dev = std_dev

    def __call__(self, img):
        img = img - img.mean()
        img = img / self.std_dev
        return img


class PairedTransform:
    """Apply identical transforms to paired images (SAR and optical).

    This ensures that both images in a pair receive the same random transformations
    (e.g., same crop location, same flip direction) to maintain spatial correspondence.
    """
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, img1, img2):
        # Get random seed for reproducible transforms
        seed = torch.randint(0, 2**32, (1,)).item()

        # Apply same transform to both images using the same random seed
        torch.manual_seed(seed)
        img1_t = self.base_transform(img1)
        torch.manual_seed(seed)
        img2_t = self.base_transform(img2)

        return img1_t, img2_t
    

def build_loader(dataset_name=None, **kwargs):
    """Builds and returns the training data loader."""
    base_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.data.img_size, scale=(0.2, 1.0), interpolation=3),
        transforms.Resize((config.data.img_size, config.data.img_size)),

        # transforms.RandomAffine(degrees=(-10,10), shear=(0, 0, -10, 10)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(contrast=0.5),

        # NormalizeSAR(std_dev=config.data.dataset_std_dev),
    ])

    if dataset_name is None:
        raise ValueError("Dataset name must be provided")
    elif dataset_name == "capella":
        # Capella: single SAR images, use base transform
        train_dataset = CapellaDataset(folder=config.data.train_data, transform=base_transform)
    elif dataset_name == "sentinel":
        # Sentinel: paired SAR-optical images, use paired transform
        paired_transform = PairedTransform(base_transform)
        train_dataset = SentinelDataset(data_path=config.data.train_data, transform=paired_transform, **kwargs)
    else:
        raise ValueError(f"Dataset: {dataset_name} not implemented yet.")

    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True,
        persistent_workers=True, pin_memory=True, num_workers=config.data.num_workers, drop_last=True)

    return train_loader