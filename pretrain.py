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

    L.seed_everything(config.train.seed, workers=True)
    
    train_loader = build_loader()

    # --- Training Run ---
    if config.model.resume is not None: # Load from a lightning checkpoint
        print(">>> Load SARATR-X model from checkpoint:", config.model.resume)
        autoencoder = SARATRX.load_from_checkpoint(config.model.resume, map_location="cpu")
    else:
        autoencoder = SARATRX(
            img_size=config.data.img_size,
            in_chans=config.model.in_chans,
            mgf_kens=config.model.mgf_kens,
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
