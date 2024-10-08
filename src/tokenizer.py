import random
from pathlib import Path
from uuid import uuid4
import filecmp
import tempfile
import torch
from omegaconf import OmegaConf, DictConfig

from miditok import REMI, MIDILike, TSD, Structured, CPWord, Octuple, MuMIDI, MMM, MusicTokenizer
from miditok.pytorch_data import DatasetMIDI, DataCollator, split_files_for_training
from torch.utils.data import random_split, Subset, Dataset


TOKENIZER_DIR = Path("tokenizers")
TOKENIZER_PARAMS_FILENAME = "params.json"
TOKENIZER_CONFIG_FILENAME = "config.yaml"

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

DATA_TEMPDIR = tempfile.TemporaryDirectory()


def prepare_tokenizer_config(config: DictConfig):
    tokenizer_config = dict(config.data)
    tokenizer_config['vocab_size'] = config.model.vocab_size
    return OmegaConf.create(tokenizer_config)


def find_matching_tokenizer_path(tokenizer_config: DictConfig,
                                 tokenizer_dir: Path | str):
    potential_config_paths = Path(tokenizer_dir).glob(
        f"**/{TOKENIZER_CONFIG_FILENAME}")
    with tempfile.NamedTemporaryFile(suffix='.yaml') as f:

        OmegaConf.save(tokenizer_config, f=f.name)
        for cfg_path in potential_config_paths:
            print(filecmp.cmp(f.name, cfg_path))
            if filecmp.cmp(f.name, cfg_path):
                return cfg_path.parent / TOKENIZER_PARAMS_FILENAME
    return None


def get_uuid():
    return str(uuid4())


def get_tokenized_dataset(config: DictConfig):
    if config.data.name == "midi":
        dataset_path = Path(config.data.path) / config.data.dataset_name
        if config.data.test_train_on_one_file and config.data.test_train_on_one_file_path:
            midi_paths = [Path(config.data.test_train_on_one_file_path).resolve()]
        else:
            midi_paths = [
                path.resolve() for path in list(dataset_path.glob("**/*.mid*"))
            ]
        tokenizer = get_tokenizer(config, midi_paths)

        tempdir_path = Path(DATA_TEMPDIR.name)
        split_files_for_training(
            files_paths=midi_paths,
            tokenizer=tokenizer,
            save_dir=tempdir_path,
            max_seq_len=config.data.max_seq_len,
        )
        dataset = DatasetMIDI(
            files_paths=list(tempdir_path.glob("**/*.mid*")),
            tokenizer=tokenizer,
            max_seq_len=config.data.max_seq_len,
            bos_token_id=tokenizer["BOS_None"],
            eos_token_id=tokenizer["EOS_None"],
        )
        print("TOKERN PAD ID", tokenizer.pad_token_id)
        collator = DataCollator(tokenizer.pad_token_id,
                                shift_labels=True,
                                copy_inputs_as_labels=True)

        dataset_size = len(dataset)
        train_size = int(config.data.train_split * dataset_size)
        val_size = dataset_size - train_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        if config.data.test_train_on_one_file:
            index_to_train = random.randint(0, dataset_size - 1)
            single_item_dataset = Subset(dataset, [index_to_train])
            print(
                f"Processing: {len(single_item_dataset)} with sequence length {len(single_item_dataset[0]['input_ids'])}"
            )
            return single_item_dataset, single_item_dataset, collator
        return train_dataset, val_dataset, collator
    else:
        collator = DataCollator(0,
                                shift_labels=True,
                                copy_inputs_as_labels=True)
        train_dataset = RandomSequenceDataset(config.data.length, config.data.size, config.data.count)
        return train_dataset, train_dataset, collator


def train_tokenizer(tokenizer_config: DictConfig, midi_paths):
    tokenizer = TOKENIZR_MAPPING[tokenizer_config.tokenizer.type.lower()]()
    tokenizer.train(vocab_size=tokenizer_config.vocab_size,
                    files_paths=midi_paths,
                    model=tokenizer_config.tokenizer.training_model)

    uuid = get_uuid()
    tokenizer.save_params(TOKENIZER_DIR / uuid / TOKENIZER_PARAMS_FILENAME)
    OmegaConf.save(tokenizer_config,
                   TOKENIZER_DIR / uuid / TOKENIZER_CONFIG_FILENAME)
    return tokenizer


def get_tokenizer(config: DictConfig,
                  midi_paths=None) -> MusicTokenizer | None:
    tokenizer_config = prepare_tokenizer_config(config)
    tokenizer_path = find_matching_tokenizer_path(tokenizer_config,
                                                  TOKENIZER_DIR)
    if tokenizer_path is not None:
        return TOKENIZR_MAPPING[config.data.tokenizer.type.lower()](
            params=tokenizer_path)

    if midi_paths is not None:
        return train_tokenizer(tokenizer_config, midi_paths)
    raise Exception("Tokenizer is tokenizer path and midi path")  # pylint: disable=broad-exception-raised


def generate_random_sequence(sequence_length, vocab_size):
    """
    Generates a random sequence of integers within a specified vocabulary size.

    Parameters:
    - sequence_length (int): The length of the sequence to generate.
    - vocab_size (int): The number of unique symbols in the vocabulary.

    Returns:
    - torch.Tensor: A tensor containing the random sequence.
    """
    # Check inputs
    if sequence_length <= 0 or vocab_size <= 0:
        raise ValueError("sequence_length and vocab_size must be positive integers")

    # Generate a random sequence of indices from 0 to vocab_size-1
    sequence = torch.randint(low=0, high=vocab_size, size=(sequence_length,))
    return sequence


class RandomSequenceDataset(Dataset):
    def __init__(self, length, size, count=1):
        self.sequence_add = []
        self.sequence = []
        for i in range(count):
            self.sequence_add.append({"attention_mask": generate_random_sequence(length, size)})
            self.sequence.append({"input_ids": generate_random_sequence(length, size)})

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, index):
        # Assuming the sequence is 1D, each item is a single token
        return self.sequence[index]
