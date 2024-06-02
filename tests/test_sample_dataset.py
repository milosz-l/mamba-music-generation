import hydra
from omegaconf import DictConfig

from tokenizer import get_tokenized_dataloader, load_pretrained_tokenizer
from utils import export_to_wav


# pylint: disable = unused-variable
@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def main(config: DictConfig):
    train_dataset, val_dataset, collator = get_tokenized_dataloader(config)

    # print first 10 and then random 20
    print ("First 10")
    for i in range(10):
        print(f"train_dataset[{i}]: {train_dataset[i]}")
    print("\nRandom 20")
    for i in range(20):
        random_index = random.randint(10, len(train_dataset) - 1)
        print(f"train_dataset[{random_index}]: {train_dataset[random_index]}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
