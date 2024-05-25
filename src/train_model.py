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
import wandb
from callbacks import get_callbacks
from pathlib import Path

@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def train_model(config: DictConfig):
    
    wandb_logger = WandbLogger(project=config.wandb.project, log_model="all")

    torch.set_float32_matmul_precision('medium')

    callbacks = get_callbacks(config.training.callbacks)

    interface_model = LighteningMamba(config)
    trainer = pl.Trainer(callbacks=callbacks, max_epochs=config.training.epochs, logger=wandb_logger)
    trainer.fit(interface_model)

    # # Save the trained model with the same name as the wandb experiment
    # model_path = Path(config.models.save_path)
    # model_path.mkdir(parents=True, exist_ok=True)
    # experiment_name = wandb.run.name
    # torch.save(interface_model.model.state_dict(), model_path / f"{experiment_name}_model.pt")

    wandb.finish()


if __name__ == "__main__":
    train_model()
