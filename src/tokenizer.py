from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator
from torch.utils.data import DataLoader
from pathlib import Path
from omegaconf import DictConfig


TOKENIZR_MAPPING = {
    'remi': REMI
}


def get_tokenized_dataloader(config: DictConfig):

    tokenizer = TOKENIZR_MAPPING[config.data.tokenizer.lower()]()
    dataset_path = Path(config.data.path) / config.data.dataset_name
    midi_paths = list(dataset_path.glob("**/*.mid*"))
    tokenizer.train(vocab_size=30000, files_paths=midi_paths)
    tokenizer.save_params(dataset_path / "tokenizer.json")

    # Split MIDIs into smaller chunks for training
    # dataset_chunks_dir = Path("path", "to", "midi_chunks")
    # split_midis_for_training(
    #     files_paths=midi_paths,
    #     tokenizer=tokenizer,
    #     save_dir=dataset_chunks_dir,
    #     max_seq_len=1024,
    # )

    # Create a Dataset, a DataLoader and a collator to train a model
    dataset = DatasetMIDI(
        files_paths=midi_paths,
        tokenizer=tokenizer,
        max_seq_len=1024,
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"],
    )
    collator = DataCollator(tokenizer.pad_token_id)
    dataloader = DataLoader(dataset, batch_size=config.model.batch_size, collate_fn=collator)
    return dataloader
