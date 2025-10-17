import os
import timm
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning as L

# temp (for test)
import torch.multiprocessing as mp
from torchvision import datasets
from torch.utils import data

from model.hivit_mae import HiViTMaskedAutoencoder
from model.mgf import MGF

SEED = torch.Generator().manual_seed(42)


# --- Model & Training Loop ---
class SARATRX(L.LightningModule):
    def __init__(self, img_size=224, patch_size=16, in_chans=1, 
                 hifeat=False, mask_ratio=0.75, mgf_kens = [9, 13, 17], **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = HiViTMaskedAutoencoder(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=in_chans, 
            out_chans=len(mgf_kens),
            hifeat=hifeat,
            **kwargs
        )
        self.guide_signal = MGF(mgf_kens)
        self.mask_ratio = mask_ratio

    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        # Forward model
        x, mask, ids_restore = self.model.forward_encoder(imgs, mask_ratio=self.mask_ratio)
        _, recon = self.model.forward_decoder(x, ids_restore)
        # Apply MGF to target
        target = self.guide_signal(imgs)
        target = self.model.patchify(target)
        # Compute loss
        num_preds = mask.sum()
        if self.model.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        loss = (recon - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / num_preds
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, _ = batch
        # Forward model
        x, mask, ids_restore = self.model.forward_encoder(imgs, mask_ratio=self.mask_ratio)
        _, recon = self.model.forward_decoder(x, ids_restore)
        # Apply MGF to target
        target = self.guide_signal(imgs)
        target = self.model.patchify(target)
        # Compute loss
        num_preds = mask.sum()
        if self.model.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        loss = (recon - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / num_preds
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    

if __name__ == "__main__":
    # --- Data ---
    transform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_set = datasets.MNIST(root="MNIST", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="MNIST", train=False, download=True, transform=transform)

    train_len = int(len(train_set) * 0.8)
    val_len = len(train_set) - train_len
    train_set, val_set = data.random_split(train_set, [train_len, val_len], generator=SEED)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=1)


    # --- Training Run ---
    autoencoder = SARATRX()
    trainer = L.Trainer(devices="auto")

    mp.set_start_method("spawn", force=True) # Avoid errors on MacOS
    trainer.fit(autoencoder, train_loader, val_loader)
