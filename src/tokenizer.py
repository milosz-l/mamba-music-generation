from pathlib import Path

from miditok import REMI, MIDILike, TSD, Structured, CPWord, Octuple, MuMIDI, MMM
from miditok.pytorch_data import DatasetMIDI, DataCollator
from torch.utils.data import random_split
from omegaconf import DictConfig
from miditok.pytorch_data import split_files_for_training
import shutil


TOKENIZR_MAPPING = {
    "remi": REMI,
    "midilike": MIDILike,
    "tsd": TSD,
    "structured": Structured,
    "cpword": CPWord,
    "octuple": Octuple,
    "mumidi": MuMIDI,
    "mmm": MMM
}


def get_tokenizer_path(config: DictConfig):
    dataset_path = Path(config.data.path) / config.data.dataset_name
    return  dataset_path / config.data.tokenizer.type / config.data.tokenizer.training_model / "params.json"


def get_tokenized_dataset(config: DictConfig):

    dataset_chunks_dir = dataset_path / Path('midi_chunks')
    if dataset_chunks_dir.exists():
        shutil.rmtree(dataset_chunks_dir)

    dataset_path = Path(config.data.path) / config.data.dataset_name
    tokenizer_path = get_tokenizer_path(config)

    if config.train.test_train_on_one_file:
        midi_paths = [Path(config.train.test_train_on_one_file_path).resolve()]
    else:
        midi_paths = [path.resolve() for path in list(dataset_path.glob("**/*.mid*"))]
    if not tokenizer_path.exists():
        tokenizer = TOKENIZR_MAPPING[config.data.tokenizer.type.lower()]()
        tokenizer.train(vocab_size=config.model.vocab_size,
                        files_paths=midi_paths,
                        model=config.data.tokenizer.training_model)
        tokenizer.save_params(tokenizer_path)
    else:
        tokenizer = TOKENIZR_MAPPING[config.data.tokenizer.type.lower()](params=tokenizer_path)


    split_files_for_training(
        files_paths=midi_paths,
        tokenizer=tokenizer,
        save_dir=dataset_chunks_dir,
        max_seq_len=config.data.max_seq_len,
    )

    dataset = DatasetMIDI(
        files_paths=list(dataset_chunks_dir.glob("**/*.mid*")),
        tokenizer=tokenizer,
        max_seq_len=config.data.max_seq_len,
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"],
    )
    collator = DataCollator(tokenizer.pad_token_id, shift_labels=True, copy_inputs_as_labels=True)

    dataset_size = len(dataset)
    train_size = int(config.data.train_split * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    if config.train.test_train_on_one_file:
        return dataset, dataset, collator
    return train_dataset, val_dataset, collator


def load_pretrained_tokenizer(config: DictConfig):
    tokenizer_path = get_tokenizer_path(config)

    tokenizer = TOKENIZR_MAPPING[config.data.tokenizer.type.lower()](params=tokenizer_path)
    return tokenizer
