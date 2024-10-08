"""
This is the demo code that uses hydra to access the parameters in under the directory config.

Author: Khuyen Tran
"""
import os

import hydra
import torch
from omegaconf import DictConfig
import traceback

from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from training_interface import LighteningMamba
from callbacks import get_callbacks
import wandb
from wandb_cleanup import cleanup_wandb_local_cache
from utils import generate_music, compare_sequences
from tokenizer import DATA_TEMPDIR

os.environ["TOKENIZERS_PARALLELISM"] = "true"


@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def train_model(config: DictConfig):

    wandb_logger = WandbLogger(project=config.wandb.project,
                               entity=config.wandb.entity,
                               group=config.wandb.group,
                               log_model=False)
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('medium')

    callbacks = get_callbacks(config)

    interface_model = LighteningMamba(config)
    trainer = pl.Trainer(callbacks=callbacks,
                         max_epochs=config.training.epochs,
                         logger=wandb_logger)
    try:
        trainer.fit(interface_model)
    except Exception:
        DATA_TEMPDIR.cleanup()
        traceback.print_exc()


    # Save the trained model with the same name as the wandb experiment locally after whole training
    #(wandb logging is handled in callback)
    # model_path = Path(config.models.save_path)
    # model_path.mkdir(parents=True, exist_ok=True)
    # experiment_name = wandb.run.name
    # torch.save(interface_model.model.state_dict(),
    #            model_path / f"{experiment_name}_model.pt")

    interface_model.model.eval()
    interface_model.model.inference_mode = True

    input_ids = torch.tensor([[interface_model.first_token]], dtype=torch.long)
    if config.data.name == "sequence":
        base_sequence = interface_model.train_dataset[0]["input_ids"]
        compare_sequences(input_ids=input_ids, model=interface_model.model, config=config, base_sequence=base_sequence)
    elif config.data.name == "midi":
        overtrained_song = None
        if config.data.test_train_on_one_file:
            overtrained_song = interface_model.train_dataset[0]["input_ids"]
        generate_music(input_ids=input_ids,
                       model=interface_model.model,
                       config=config,
                       overtrained_song=overtrained_song)
    wandb.finish()

    cleanup_wandb_local_cache()


if __name__ == "__main__":
    train_model()  # pylint: disable=no-value-for-parameter
