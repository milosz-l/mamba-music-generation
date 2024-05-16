"""
This is the demo code that uses hydra to access the parameters in under the directory config.

Author: Khuyen Tran
"""

import hydra
from omegaconf import DictConfig

from tokenizer import get_tokenized_dataloader


@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def train_model(config: DictConfig):

    dataloader = get_tokenized_dataloader(config)

    for batch in dataloader:
        print(batch)

        

if __name__ == "__main__":
    train_model()
