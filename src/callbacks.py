from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl


def get_callbacks(callback_config):
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.0,
        patience=10,
        verbose=False,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(
        filename='train',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )
    return [early_stop_callback, checkpoint_callback]