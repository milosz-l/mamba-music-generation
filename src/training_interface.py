import torch
import torch.nn as nn
from omegaconf import DictConfig


import pytorch_lightning as pl
from tokenizer import get_tokenized_dataloader
from mamba_model import get_mamba_model
from torch.utils.data import DataLoader


class LighteningMamba(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super(LighteningMamba, self).__init__()
        self.config = config
        self.num_workers = config.training.num_workers
        self.batch_size = config.training.batch_size
        self.learning_rate = config.training.learning_rate

        self.model = get_mamba_model(config.model)
        self.loss_function = nn.CrossEntropyLoss()
        self.embed = torch.nn.Embedding(
            num_embeddings=config.model.vocab_size,
            embedding_dim=config.model.model_dimension
        )
        self.train_dataset, self.val_dataset, self.collator = get_tokenized_dataloader(config.data)

        self.save_hyperparameters(ignore=['model'])

    def forward(self, input_ids, attention_mask=None):
        embeddings = self.embed(input_ids)
        return self.model(embeddings)

    def predict(self, input_ids, attention_mask=None):
        embeddings = self.embed(input_ids)
        return self.model(embeddings)

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        outputs = self(input_ids, attention_mask)

        loss = self.loss_function(outputs.transpose(1, 2), input_ids)  # Adjust dimensions if necessary

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        outputs = self(input_ids, attention_mask)

        loss = self.loss_function(outputs.transpose(1, 2), input_ids)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss}

    def on_validation_epoch_end(self):
        pass

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.training.batch_size, collate_fn=self.collator)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.training.batch_size, collate_fn=self.collator)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        return [optimizer], [scheduler]
