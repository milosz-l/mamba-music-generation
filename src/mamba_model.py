import math
from functools import partial
from omegaconf import DictConfig

import torch
from torch import nn

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MixerModel
from mamba_ssm.utils.generation import GenerationMixin


def get_mamba_model(config: DictConfig):
    mamba_config = MambaConfig(
        d_model=config.model_dimension,
        n_layer=config.n_layer,
        vocab_size=config.vocab_size,
        rms_norm=config.rms_norm,
        residual_in_fp32=config.residual_in_fp32,
        fused_add_norm=config.fused_add_norm,
        pad_vocab_size_multiple=config.pad_vocab_size_multiple,
        tie_embeddings=config.tie_embeddings,
        ssm_cfg=config.ssm_config)
    return MambaMusicHead(mamba_config)


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


# class LogitsWrapper(torch.Tensor):
#     def __init__(self, logits):
#         super().__init__()
#         self.logits = logits


class MambaMusicHead(nn.Module, GenerationMixin):

    def __init__(
        self,
        config: MambaConfig,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        self.config = config
        d_model = config.d_model
        n_layer = config.n_layer
        vocab_size = config.vocab_size
        ssm_cfg = config.ssm_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        fused_add_norm = config.fused_add_norm
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size %
                                                     pad_vocab_size_multiple)
        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(d_model,
                                 vocab_size,
                                 bias=False,
                                 **factory_kwargs)

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            ))
        self.tie_weights()

    def tie_weights(self):
        if self.config.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(self,
                                 batch_size,
                                 max_seqlen,
                                 dtype=None,
                                 **kwargs):
        return self.backbone.allocate_inference_cache(batch_size,
                                                      max_seqlen,
                                                      dtype=dtype,
                                                      **kwargs)

    def forward(self, input_ids, **kwargs):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.backbone(input_ids, inference_params=None)
        # if num_last_tokens > 0:
        #     hidden_states = hidden_states[:, -num_last_tokens:]
        return self.lm_head(hidden_states)

    # @classmethod
    # def from_pretrained(cls, pretrained_model_name, device=None, dtype=None, **kwargs):
    #     config_data = load_config_hf(pretrained_model_name)
    #     config = MambaConfig(**config_data)
    #     model = cls(config, device=device, dtype=dtype, **kwargs)
    #     model.load_state_dict(load_state_dict_hf(pretrained_model_name, device=device, dtype=dtype))
    #     return model

    # def save_pretrained(self, save_directory):
    #     """
    #     Minimal implementation of save_pretrained for MambaLMHeadModel.
    #     Save the model and its configuration file to a directory.
    #     """
    #     # Ensure save_directory exists
    #     os.makedirs(save_directory, exist_ok=True)
    #
    #     # Save the model's state_dict
    #     model_path = os.path.join(save_directory, 'pytorch_model.bin')
    #     torch.save(self.state_dict(), model_path)
    #
    #     # Save the configuration of the model
    #     config_path = os.path.join(save_directory, 'config.json')
    #     with open(config_path, 'w') as f:
    #         json.dump(self.config.__dict__, f)
