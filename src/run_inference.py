import torch
import hydra
from omegaconf import DictConfig
from mamba_model import get_mamba_model
import wandb
from pathlib import Path

def load_model(model_path, config):
    model = get_mamba_model(config.model)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def run_inference(model, input_ids):
    with torch.no_grad():
        output = model(input_ids)
    return output

@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def main(config: DictConfig):
    if config.inference.wandb_run_id:
        # Initialize wandb and download the model file
        wandb.init(project=config.wandb.project, id=config.inference.wandb_run_id, resume="allow")
        model_path = wandb.restore('model.pt').name
    else:
        model_path = config.inference.model_path

    model = load_model(model_path, config)
    
    # Example input_ids, replace with your actual input data
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    
    output = run_inference(model, input_ids)
    print(output)

if __name__ == "__main__":
    main()