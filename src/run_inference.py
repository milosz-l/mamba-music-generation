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
    model = get_mamba_model(config.model)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model.to('cuda')  # Move model to GPU

def generate_sequence(model, input_ids, max_length, temperature=0.8, top_k=50):
    model.eval()
    generated = input_ids.to('cuda')
    for _ in range(max_length - input_ids.size(1)):
        with torch.no_grad():
            print(f"Generated: {generated}")
            output = model(generated)
            # print output, output shape, output length, top 5 max and min values, etc.
            print(f"Output: {output}")
            print(f"Output shape: {output.shape}")
            print(f"Output length: {output.size(1)}")
            print(f"Output max value: {torch.max(output)}")
            print(f"Output min value: {torch.min(output)}")
            next_token_logits = output[:, -1, :] / temperature
            filtered_logits = top_k_filtering(next_token_logits, top_k=top_k)
            next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
            print(f"Next token logits: {next_token_logits}")
            print(f"Filtered logits: {filtered_logits}")
            print(f"Next token: {next_token}")
    return generated

def top_k_filtering(logits, top_k=50):
    if top_k > 0:
        values, indices = torch.topk(logits, top_k)
        min_values = values[:, -1, None]
        logits = torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)
    return logits

@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def main(config: DictConfig):
    if config.inference.wandb_run_id:
        # TODO: test it
        print("taking model from wandb_run_id")
        # Initialize wandb and download the model file
        wandb.init(project=config.wandb.project, entity=config.wandb.entity, id=config.inference.wandb_run_id, resume="allow")
        model_path = wandb.restore('model.pt').name
    else:
        print("taking model from path")
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
    generated_sequence = generate_sequence(model, input_ids, max_length, temperature, top_k)
    print(f"generated_sequence: {generated_sequence}")
    print(f"generated_sequence.shape: {generated_sequence.shape}")
    
    # Export the output to a .wav file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = model_path.split("-")[-2]
    wav_filename = f'output_{model_name}_{timestamp}.wav'

    # Load the tokenizer
    tokenizer = load_pretrained_tokenizer(config)
    
    # Ensure the output is in the correct format for the tokenizer
    tokens = generated_sequence.cpu().numpy()
    
    export_to_wav(tokenizer, tokens, wav_filename)


if __name__ == "__main__":
    main()