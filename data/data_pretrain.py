from torch.utils.data import DataLoader
from torchvision import transforms

import config

from ..data import CapellaDataset, SentinelDataset


# TODO: check normalization values for sentinel-1 and sentinel-2
class NormalizeSAR:
    """SAR Normalization used in TRANSAR paper."""
    def __init__(self, std_dev):
        self.std_dev = std_dev

    def __call__(self, img):
        img = img - img.mean()
        img = img / self.std_dev
        return img
    

def build_loader(dataset_name=None, **kwargs):
    """Builds and returns the training data loader."""
    transform = transforms.Compose([
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
        train_dataset = CapellaDataset(folder=config.data.train_data, transform=transform)
    elif dataset_name == "sentinel":
        train_dataset = SentinelDataset(data_path=config.data.train_data, transform=transform, **kwargs)
    else:
        raise ValueError(f"Dataset: {dataset_name} not implemented yet.")

    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, 
        persistent_workers=True, pin_memory=True, num_workers=config.data.num_workers, drop_last=True)

    return train_loader