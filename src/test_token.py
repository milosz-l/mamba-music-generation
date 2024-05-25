from tokenizer import get_tokenized_dataloader, load_pretrained_tokenizer
from omegaconf import DictConfig
import hydra
from utils import export_to_wav


@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def main(config: DictConfig):
    train_dataset, val_dataset, collator = get_tokenized_dataloader(config)
    tokenizer = load_pretrained_tokenizer(config)
    print(train_dataset[0]['input_ids'].unsqueeze(0))
    print(tokenizer.is_trained)
    export_to_wav(tokenizer, train_dataset[0]['input_ids'].unsqueeze(0), 'out.wav')


if __name__ == "__main__":
    main()