import platform

import torch.multiprocessing as mp
import lightning as L
from lightning.pytorch.callbacks import ModelSummary

from config import config
from data.data_pretrain import build_loader
from model.saratrx import SARATRX


if __name__ == "__main__":
    if platform.system() == "Darwin":
        mp.set_start_method("spawn", force=True) # Avoid errors on MacOS

    L.seed_everything(config.TRAIN.SEED, workers=True)

    # Validate config consistency
    if config.MODEL.TARGET_MODE == "optical" and config.DATA.DATASET_NAME != "sentinel":
        raise ValueError(
            f"target_mode='optical' requires dataset_name='sentinel', "
            f"got dataset_name='{config.DATA.DATASET_NAME}'"
        )

    train_loader = build_loader(dataset_name=config.DATA.DATASET_NAME) # NOTE: when using sentinel, can pass terrains argument here

    # --- Training Run ---
    if config.MODEL.RESUME is not None: # Load from a lightning checkpoint
        print(">>> Load SARATR-X model from checkpoint:", config.MODEL.RESUME)
        autoencoder = SARATRX.load_from_checkpoint(config.MODEL.RESUME, map_location="cpu")
    else:
        autoencoder = SARATRX(
            img_size=config.DATA.IMG_SIZE,
            in_chans=config.MODEL.IN_CHANS,
            mgf_kens=config.MODEL.MGF_KENS,
            target_mode=config.MODEL.TARGET_MODE,
            norm_pix_loss=config.MODEL.NORM_PIX_LOSS,
        )

    trainer = L.Trainer(
        callbacks=ModelSummary(max_depth=0),
        gradient_clip_val=config.TRAIN.CLIP_GRAD,
        precision="16-mixed",
        devices=config.TRAIN.N_GPU,
        accelerator="auto",
        max_epochs=config.TRAIN.EPOCHS,
        log_every_n_steps=50,
        deterministic=True,
        enable_progress_bar=True,
    )
    trainer.fit(autoencoder, train_loader)
