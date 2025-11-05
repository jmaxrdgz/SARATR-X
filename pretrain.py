import math
import platform

import torch
from torch import nn
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import LambdaLR
import lightning as L
from lightning.pytorch.callbacks import ModelSummary
from functools import partial

from config import config
from data.data_pretrain import build_loader
from model.hivit_mae import HiViTMaskedAutoencoder
from model.mgf import MGF


class SARATRX(L.LightningModule):
    def __init__(self, img_size=512, mask_ratio=0.75, mgf_kens = [9, 13, 17], **kwargs):
        super().__init__()
        self.example_input_array = torch.Tensor(
            config.train.batch_size, config.model.in_chans, config.data.img_size, config.data.img_size)
        self.save_hyperparameters()

        self.model = HiViTMaskedAutoencoder(
            img_size=img_size, embed_dim=512, depths=[2, 2, 20], num_heads=8, 
            stem_mlp_ratio=3., mlp_ratio=4., decoder_embed_dim=512, decoder_depth=6, decoder_num_heads=16, 
            hifeat=True, rpe=False, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

        # TODO: Convert to single dim input without breaking pretrained weights loading (average first conv weights)
        # Load pretrained weights
        if config.model.resume is None:
            state_dict = torch.load(
                config.train.init_weights, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict, strict=False)
            print(">>> Load pretrained ImageNet weights")
        
        self.guide_signal = MGF(mgf_kens)
        self.mask_ratio = mask_ratio


    def training_step(self, batch, batch_idx):
        imgs, _ = batch

        latent, mask, ids_restore = self.model.forward_encoder(imgs, self.mask_ratio)
        cls_pred, pred = self.model.forward_decoder(latent, ids_restore)

        loss = self._forward_loss(imgs, cls_pred, pred, mask)

        self.log("train_loss", loss, prog_bar=True)
        return loss


    def _forward_loss(self, imgs, cls_pred, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        num_preds = mask.sum()

        # Apply MGF to target
        mgf = self.guide_signal(imgs[:, 0:1, :, :])
        targets = [self.model.patchify(m) for m in mgf]
        target = torch.cat(targets, dim=-1)

        if self.model.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / num_preds
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=float(config.train.lr),
            weight_decay=config.train.weight_decay,
            betas=(config.train.optimizer_momentum.beta1, config.train.optimizer_momentum.beta2),
        )

        # Cosine LR wcheduler with warmup
        def lr_lambda(current_epoch):
            if current_epoch < config.train.warmup_epochs:
                return float(current_epoch) / float(max(1, config.train.warmup_epochs))
            progress = (current_epoch - config.train.warmup_epochs) / float(
                max(1, config.train.epochs - config.train.warmup_epochs)
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = {
            "scheduler": LambdaLR(optimizer, lr_lambda=lr_lambda),
            "interval": "epoch",   # or "step"
            "frequency": 1,
        }

        return [optimizer], [scheduler]


if __name__ == "__main__":
    if platform.system() == "Darwin":
        mp.set_start_method("spawn", force=True) # Avoid errors on MacOS

    L.seed_everything(config.train.seed, workers=True)

    # Validate config consistency
    if config.model.target_mode == "optical" and config.data.dataset_name != "sentinel":
        raise ValueError(
            f"target_mode='optical' requires dataset_name='sentinel', "
            f"got dataset_name='{config.data.dataset_name}'"
        )

    train_loader = build_loader(dataset_name=config.data.dataset_name) # NOTE: when using sentinel, can pass terrains argument here

    # --- Training Run ---
    if config.model.resume is not None: # Load from a lightning checkpoint
        print(">>> Load SARATR-X model from checkpoint:", config.model.resume)
        autoencoder = SARATRX.load_from_checkpoint(config.model.resume, map_location="cpu")
    else:
        autoencoder = SARATRX(
            img_size=config.data.img_size,
            in_chans=config.model.in_chans,
            mgf_kens=config.model.mgf_kens,
            target_mode=config.model.target_mode,
            norm_pix_loss=config.model.norm_pix_loss,
        )

    trainer = L.Trainer(
        callbacks=ModelSummary(max_depth=0),
        gradient_clip_val=config.train.clip_grad,
        precision="16-mixed",
        devices=config.train.n_gpu,
        accelerator="auto",
        max_epochs=config.train.epochs,
        log_every_n_steps=50,
        deterministic=True,
        enable_progress_bar=True,
    )
    trainer.fit(autoencoder, train_loader)
