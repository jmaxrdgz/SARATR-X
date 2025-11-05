import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import lightning as L
from functools import partial

from config import config
from .hivit_mae import HiViTMaskedAutoencoder
from .mgf import MGF


class SARATRX(L.LightningModule):
    def __init__(self, img_size=512, mask_ratio=0.75, mgf_kens = [9, 13, 17], target_mode="mgf", **kwargs):
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
                "checkpoints/mae_hivit_base_1600ep.pth", map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict, strict=False)
            print(">>> Load pretrained ImageNet weights")

        self.guide_signal = MGF(mgf_kens)
        self.mask_ratio = mask_ratio
        self.target_mode = target_mode

        # Validate target_mode
        assert target_mode in ["mgf", "optical"], \
            f"target_mode must be 'mgf' or 'optical', got '{target_mode}'"


    def training_step(self, batch, batch_idx):
        imgs, target_imgs = batch

        latent, mask, ids_restore = self.model.forward_encoder(imgs, self.mask_ratio)
        cls_pred, pred = self.model.forward_decoder(latent, ids_restore)

        loss = self._forward_loss(imgs, target_imgs, cls_pred, pred, mask)

        self.log("train_loss", loss, prog_bar=True)
        return loss


    def _forward_loss(self, imgs, target_imgs, cls_pred, pred, mask):
        """
        imgs: [N, 3, H, W] - Input images (SAR)
        target_imgs: [N, 3, H, W] - Target images (optical for Sentinel, or placeholder for Capella)
        pred: [N, L, p*p*C]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        num_preds = mask.sum()

        if self.target_mode == "mgf":
            # Apply MGF to SAR input (original behavior for Capella, also works for Sentinel SAR)
            mgf = self.guide_signal(imgs[:, 0:1, :, :])
            targets = [self.model.patchify(m) for m in mgf]
            target = torch.cat(targets, dim=-1)
        elif self.target_mode == "optical":
            # Validate target_imgs is a proper tensor
            if not isinstance(target_imgs, torch.Tensor):
                raise TypeError(
                    f"target_imgs must be a Tensor in optical mode, "
                    f"got {type(target_imgs)}"
                )
            if target_imgs.shape != imgs.shape:
                raise ValueError(
                    f"Target shape {target_imgs.shape} must match "
                    f"input shape {imgs.shape}"
                )
            # Use optical image directly as target (for Sentinel dataset)
            target = self.model.patchify(target_imgs)
        else:
            raise ValueError(f"Unknown target_mode: {self.target_mode}")

        # Validate pred and target shapes match
        if pred.shape != target.shape:
            raise RuntimeError(
                f"Prediction shape {pred.shape} doesn't match "
                f"target shape {target.shape}. This may indicate a mismatch between "
                f"the decoder output channels and the target representation."
            )

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
            return 0.5 * (1.0 + torch.cos(torch.pi * progress))

        scheduler = {
            "scheduler": LambdaLR(optimizer, lr_lambda=lr_lambda),
            "interval": "epoch",   # or "step"
            "frequency": 1,
        }

        return [optimizer], [scheduler]