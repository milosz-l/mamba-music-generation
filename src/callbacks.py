from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def get_callbacks():
    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        min_delta=0.0,
                                        patience=10,
                                        verbose=False,
                                        mode='min')

    checkpoint_callback = ModelCheckpoint(
        filename='best-checkpoint-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )

    # periodic_checkpoint_callback = ModelCheckpoint(
    #     filename='checkpoint-{epoch:02d}-{val_loss:.2f}',
    #     save_top_k=-1,  # Save all checkpoints
    #     every_n_epochs=5,  # Save every 5 epochs
    #     verbose=True,
    # )

    return [
        early_stop_callback, checkpoint_callback    #, periodic_checkpoint_callback
    ]
