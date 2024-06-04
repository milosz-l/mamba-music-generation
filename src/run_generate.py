import os
import torch
import hydra
from omegaconf import DictConfig
from mamba_model import get_mamba_model
import wandb
from pathlib import Path
from symusic import Synthesizer, dump_wav
from datetime import datetime
from utils import export_to_wav
from tokenizer import load_pretrained_tokenizer  # Import the tokenizer

def load_model(model_path, config):
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
    return model.to('cuda')  # Move model to GPU

@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def main(config: DictConfig):
    if config.inference.wandb_model_full_name:
        print(f"taking this model from wandb: {config.inference.wandb_model_full_name}")
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

    print(f"input: {input_ids}")
    max_length = config.inference.max_length
    temperature = config.inference.get('temperature', 0.8)
    top_k = config.inference.get('top_k', 50)
    
    # Use model.generate for sequence generation
    generated_sequence = model.generate(
        input_ids=input_ids.to('cuda'),
        max_length=max_length,
        temperature=temperature,
        top_k=top_k
    )
    
    print(f"generated_sequence: {generated_sequence}")
    print(f"generated_sequence.shape: {generated_sequence.shape}")
    
    # Export the output to a .wav file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wav_filename = f'output_{timestamp}.wav'

    # Load the tokenizer
    tokenizer = load_pretrained_tokenizer(config)
    
    # Ensure the output is in the correct format for the tokenizer
    tokens = generated_sequence.cpu().numpy()
    
    export_to_wav(tokenizer, tokens, wav_filename)


if __name__ == "__main__":
    main()