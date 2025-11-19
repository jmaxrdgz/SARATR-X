import math
import torch
from torch import nn
from torch.optim import AdamW
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
            config.TRAIN.BATCH_SIZE, config.MODEL.IN_CHANS, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)
        self.save_hyperparameters()

        self.model = HiViTMaskedAutoencoder(
            img_size=img_size, embed_dim=512, depths=[2, 2, 20], num_heads=8,
            stem_mlp_ratio=3., mlp_ratio=4., decoder_embed_dim=512, decoder_depth=6, decoder_num_heads=16,
            hifeat=True, rpe=False, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

        # Load pretrained weights
        if config.MODEL.RESUME is None:
            state_dict = torch.load(
                config.TRAIN.INIT_WEIGHTS, map_location="cpu", weights_only=True)

            # Adapt first conv layer from 3 channels to in_chans by averaging
            if kwargs.get('in_chans', 3) != 3 and 'patch_embed.proj.weight' in state_dict:
                conv_weight = state_dict['patch_embed.proj.weight']  # shape: [out_channels, 3, kernel_h, kernel_w]
                # Average across the 3 input channels and repeat to match in_chans
                avg_weight = conv_weight.mean(dim=1, keepdim=True)  # shape: [out_channels, 1, kernel_h, kernel_w]
                state_dict['patch_embed.proj.weight'] = avg_weight.repeat(1, kwargs.get('in_chans', 3), 1, 1)
                print(f">>> Adapted first conv layer from 3 to {kwargs.get('in_chans', 3)} channels")

            # Exclude position embeddings if image size changed (they'll be regenerated with sin-cos)
            if img_size != 224:
                excluded_keys = ['absolute_pos_embed', 'decoder_pos_embed']
                for key in excluded_keys:
                    if key in state_dict:
                        del state_dict[key]
                grid_size = img_size // 16  # patch_size = 16
                print(f">>> Excluded pos_embed from checkpoint (using sin-cos for {grid_size}x{grid_size} grid)")

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

        # Log loss and learning rate
        self.log("train_loss", loss, prog_bar=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)
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
        """Configure optimizer with cosine annealing and warmup."""
        # AdamW optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=config.TRAIN.LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY,
            betas=(config.TRAIN.OPTIMIZER_MOMENTUM.BETA1, config.TRAIN.OPTIMIZER_MOMENTUM.BETA2)
        )

        # Cosine annealing with warmup
        def lr_lambda(current_step):
            warmup = config.TRAIN.WARMUP_EPOCHS * self.trainer.num_training_batches
            total = config.TRAIN.EPOCHS * self.trainer.num_training_batches

            if current_step < warmup:
                # Linear warmup
                return float(current_step) / float(max(1, warmup))
            else:
                # Cosine annealing
                progress = float(current_step - warmup) / float(max(1, total - warmup))
                return 0.5 * (1.0 + math.cos(progress * math.pi))

        scheduler = LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # Update every step, not epoch
                'frequency': 1
            }
        }