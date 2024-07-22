import os
import torch
import hydra
from omegaconf import DictConfig
from mamba_model import get_mamba_model
import wandb

from utils import generate_music, compare_sequences


def load_model(model_path, config, inference_mode=True):
    checkpoint = torch.load(model_path)

    # Extract the state_dict from the checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remove 'model.' prefix from state_dict keys if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            new_state_dict[k[6:]] = v  # Remove 'model.' prefix
        else:
            new_state_dict[k] = v

    model = get_mamba_model(config.model)
    model.load_state_dict(new_state_dict)
    model.eval()
    model.inference_mode = inference_mode
    return model.to('cuda')  # Move model to GPU


@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def main(config: DictConfig):
    if config.inference.wandb_model_full_name:
        print(
            f"taking this model from wandb: {config.inference.wandb_model_full_name}"
        )
        # Initialize wandb and download the model file
        # run = wandb.init(project=config.wandb.project, entity=config.wandb.entity, resume="allow")
        # artifact = run.use_artifact(config.inference.wandb_model_full_name, type='model')
        # artifact_dir = artifact.download()

        # Use wandb.Api to fetch the model artifact
        api = wandb.Api()
        artifact = api.artifact(config.inference.wandb_model_full_name)

        download_dir = config.models.save_path
        model_dir = artifact.download(root=download_dir)

        model_path = os.path.join(model_dir, 'model.ckpt')
    else:
        print(f"taking model from path, model: {config.inference.model_path}")
        model_path = config.inference.model_path

    model = load_model(model_path, config)

    # Generate random input if input_ids are not provided
    if 'input_ids' in config.inference:
        input_ids = torch.tensor(config.inference.input_ids).unsqueeze(0)
    else:
        input_length = config.inference.input_length
        vocab_size = config.model.vocab_size
        input_ids = torch.randint(0, vocab_size, (1, input_length))
    overtrained_song = None
    if config.inference.overtrained_song:
        overtrained_song = config.inference.overtrained_song


    compare_sequences(input_ids, model, config, base_sequence=overtrained_song)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
