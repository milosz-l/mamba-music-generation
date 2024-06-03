from pathlib import Path

from miditok import REMI
from miditok.pytorch_data import DatasetMIDI, DataCollator
from torch.utils.data import random_split
from omegaconf import DictConfig

TOKENIZR_MAPPING = {'remi': REMI}


def get_tokenized_dataloader(config: DictConfig):

    dataset_path = Path(config.data.path) / config.data.dataset_name
    tokenizer_path = dataset_path / config.data.tokenizer / "params.json"

    midi_paths = list(dataset_path.glob("**/*.mid*"))
    if not tokenizer_path.exists():
        tokenizer = TOKENIZR_MAPPING[config.data.tokenizer.lower()]()
        tokenizer.train(vocab_size=config.model.vocab_size,
                        files_paths=midi_paths,
                        model="BPE")
        tokenizer.save_params(tokenizer_path)
    else:
        tokenizer = TOKENIZR_MAPPING[config.data.tokenizer.lower()](params=tokenizer_path)


    dataset = DatasetMIDI(
        files_paths=midi_paths,
        tokenizer=tokenizer,
        max_seq_len=1024,
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"],
    )
    collator = DataCollator(tokenizer.pad_token_id, shift_labels=True)

    dataset_size = len(dataset)
    train_size = int(config.data.train_split * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset, collator


def load_pretrained_tokenizer(config: DictConfig):
    dataset_path = Path(config.data.path) / config.data.dataset_name
    tokenizer_path = dataset_path / config.data.tokenizer / "params.json"

    tokenizer = TOKENIZR_MAPPING[config.data.tokenizer.lower()](params=tokenizer_path)
    return tokenizer
