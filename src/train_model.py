"""
This is the demo code that uses hydra to access the parameters in under the directory config.

Author: Khuyen Tran
"""

import hydra
from omegaconf import DictConfig
import torch
from mamba_ssm import Mamba

from tokenizer import get_tokenized_dataloader


@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def train_model(config: DictConfig):

    dataloader = get_tokenized_dataloader(config)

    for batch in dataloader:
        print(batch)

    batch, length, dim = 2, 64, 16
    print(torch.cuda.is_available())
    x = torch.randn(batch, length, dim).to("cuda")
    model = Mamba(
        # This module uses roughly 3 * expand * d_model^2 parameters
        d_model=dim,  # Model dimension d_model
        d_state=16,  # SSM state expansion factor
        d_conv=4,  # Local convolution width
        expand=2,  # Block expansion factor
    ).to("cuda")
    y = model(x)
    assert y.shape == x.shape
    print(f"Train modeling using {config.data.processed}")
    print(f"Model used: {config.model.name}")
    print(f"Save the output to {config.data.final}")



if __name__ == "__main__":
    train_model()
