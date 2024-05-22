from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator
from torch.utils.data import random_split
from pathlib import Path
from omegaconf import DictConfig


TOKENIZR_MAPPING = {
    'remi': REMI
}


def get_tokenized_dataloader(config: DictConfig):

    tokenizer = TOKENIZR_MAPPING[config.tokenizer.lower()]()
    dataset_path = Path(config.path) / config.dataset_name
    midi_paths = list(dataset_path.glob("**/*.mid*"))
    tokenizer.train(vocab_size=30000, files_paths=midi_paths)
    tokenizer.save_params(dataset_path / "tokenizer.json")

    dataset = DatasetMIDI(
        files_paths=midi_paths,
        tokenizer=tokenizer,
        max_seq_len=1024,
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"],
    )
    collator = DataCollator(tokenizer.pad_token_id)

    dataset_size = len(dataset)
    train_size = int(config.train_split * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset, collator
