import torch
from torch import nn
from functools import partial
import lightning as L

# temp (for test)
import torch.multiprocessing as mp

from config import config
from model.hivit_mae import HiViTMaskedAutoencoder
from model.mgf import MGF
from data.data_pretrain import build_loader


# --- Model & Training Loop ---
class SARATRX(L.LightningModule):
    def __init__(self, img_size=512, patch_size=16, in_chans=3, out_chans=3,
                 mask_ratio=0.75, mgf_kens = [9, 13, 17], **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = HiViTMaskedAutoencoder(
            img_size=img_size, embed_dim=512, depths=[2, 2, 20], num_heads=8, stem_mlp_ratio=3.,
            mlp_ratio=4., decoder_embed_dim=512, decoder_depth=6, decoder_num_heads=16, hifeat=True,
            rpe=False, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

        # TODO: Convert to single dim input without breaking pretrained weights loading (average first conv weights)
        # Load pretrained weights
        if config.model.resume is None:
            state_dict = torch.load("checkpoints/mae_hivit_base_1600ep.pth", map_location="cpu")
            self.model.load_state_dict(state_dict, strict=False)
        
        self.guide_signal = MGF(mgf_kens)
        self.mask_ratio = mask_ratio

    def training_step(self, batch, batch_idx):
        imgs, _ = batch
        # Forward model
        x, mask, ids_restore = self.model.forward_encoder(imgs, mask_ratio=self.mask_ratio)
        _, recon = self.model.forward_decoder(x, ids_restore)
        # Apply MGF to target
        target = self.guide_signal(imgs[:, 0:1, :, :])
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True) # Avoid errors on MacOS

    L.seed_everything(config.train.seed, workers=True)
    
    train_loader, val_loader = build_loader()

    # --- Training Run ---
    if config.model.resume is not None: # Load from a lightning checkpoint
        model = SARATRX.load_from_checkpoint(config.model.resume, map_location="cpu")
    else:
        model = SARATRX(img_size=config.data.img_size, in_chans=config.model.in_chans)

    trainer = L.Trainer(
        precision=16, # AMP
        devices="auto",
        accelerator="gpu",
    )
    trainer.fit(model, train_loader, val_loader)
