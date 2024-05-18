"""
This is the demo code that uses hydra to access the parameters in under the directory config.

Author: Khuyen Tran
"""

import hydra
import torch
from omegaconf import DictConfig
from training_interface import LighteningMamba
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from callbacks import get_callbacks

from tokenizer import get_tokenized_dataloader


@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def train_model(config: DictConfig):
    torch.set_float32_matmul_precision('medium')

    callbacks = get_callbacks(config.training.callbacks)

    wandb_logger = WandbLogger(log_model="all")
    interface_model = LighteningMamba(config)
    trainer = pl.Trainer(callbacks=callbacks, max_epochs=config.training.epochs, logger=wandb_logger)
    trainer.fit(interface_model)


if __name__ == "__main__":
    train_model()
