import hydra
from omegaconf import DictConfig

from src.tokenizer import get_tokenized_dataset, load_pretrained_tokenizer
from src.utils import export_to_wav


# pylint: disable = unused-variable
@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def main(config: DictConfig):
    train_dataset, val_dataset, collator = get_tokenized_dataset(config)
    print(f"train_dataset: {train_dataset}")
    print(f"train_dataset[0]: {train_dataset[0]}")
    print(f"train_dataset[0]['input_ids']: {train_dataset[0]['input_ids']}")
    print(f"train_dataset[0]['input_ids'].shape: {train_dataset[0]['input_ids'].shape}")
    print(f"train_dataset[0]['input_ids'].unsqueeze(0): {train_dataset[0]['input_ids'].unsqueeze(0)}")
    print(f"train_dataset[0]['input_ids'].unsqueeze(0).shape: {train_dataset[0]['input_ids'].unsqueeze(0).shape}")
    tokenizer = load_pretrained_tokenizer(config)
    export_to_wav(tokenizer, train_dataset[0]['input_ids'].unsqueeze(0),
                  'out.wav')


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
