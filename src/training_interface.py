import torch
from torch import nn
from torch.utils.data import DataLoader

from omegaconf import DictConfig

import pytorch_lightning as pl

from tokenizer import get_tokenized_dataloader
from mamba_model import get_mamba_model


# pylint: disable=arguments-differ
class LighteningMamba(pl.LightningModule):

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.num_workers = config.training.num_workers
        self.batch_size = config.training.batch_size
        self.learning_rate = config.training.learning_rate
        self.step_size = config.training.step_size
        self.gamma = config.training.gamma

        self.loss_function = nn.CrossEntropyLoss()
        self.model = get_mamba_model(config.model)
        self.train_dataset, self.val_dataset, self.collator = get_tokenized_dataloader(
            config)

        self.save_hyperparameters(ignore=['model'])

    def forward(self, input_ids):
        return self.model(input_ids)

    def predict(self, input_ids):
        return self.model(input_ids)

    # pylint: disable = unused-argument
    def training_step(self, batch, batch_idx):
        input_ids, _ = batch['input_ids'], batch['attention_mask']
        outputs = self(input_ids)
        loss = self.loss_function(outputs.transpose(1, 2), input_ids)

        self.log('train_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return loss

    # pylint: disable = unused-argument
    def validation_step(self, batch, batch_idx):
        input_ids, _ = batch['input_ids'], batch['attention_mask']
        outputs = self(input_ids)
        loss = self.loss_function(outputs.transpose(1, 2), input_ids)

        self.log('val_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return {'val_loss': loss}

    def on_validation_epoch_end(self):
        pass

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.config.training.batch_size,
                          collate_fn=self.collator,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.config.training.batch_size,
                          collate_fn=self.collator,
                          num_workers=self.num_workers)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=3,
                                                    gamma=0.1)
        return [optimizer], [scheduler]
