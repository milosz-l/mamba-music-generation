from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from wandb_cleanup import delete_models_witout_tags_in_wandb


class WandbCleanupCallback(Callback):

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.interval = config.wandb.cleanup.run_cleanup_every_epochs

    def on_epoch_end(self, trainer, pl_module):  # pylint: disable=unused-argument
        print("running wandb cleanup on epoch end")
        if trainer.current_epoch % self.interval == 0:
            delete_models_witout_tags_in_wandb(config=self.config)


def get_callbacks(config: DictConfig):
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0,
        patience=config.training.callbacks.patience,
        verbose=True,
        mode='min')

    checkpoint_callback = ModelCheckpoint(
        filename='best-checkpoint',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )

    wandb_cleanup_callback = WandbCleanupCallback(config)

    # periodic_checkpoint_callback = ModelCheckpoint(
    #     filename='checkpoint-{epoch:02d}-{val_loss:.2f}',
    #     save_top_k=-1,  # Save all checkpoints
    #     every_n_epochs=5,  # Save every 5 epochs
    #     verbose=True,
    # )

    return [
        early_stop_callback,
        checkpoint_callback,
        wandb_cleanup_callback  #, periodic_checkpoint_callback
    ]
