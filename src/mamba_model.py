from omegaconf import DictConfig

from mamba_ssm import Mamba


def get_mamba_model(config: DictConfig):
    return Mamba(
        d_model=config.model_dimension,  # Model dimension d_model
        d_state=config.state_expansion_factor,  # SSM state expansion factor
        d_conv=config.conv_width,  # Local convolution width
        expand=config.block_expansion_factor,  # Block expansion factor
        )