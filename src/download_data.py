from pathlib import Path
import hydra
import requests
from functools import partial
import shutil
from subprocess import call

import muspy
from omegaconf import DictConfig


def download_from_url(url, save_path, verbose=False, **kwargs):
    archivepath = save_path.with_suffix('.tar.gz')
    print(f'Downloading to {archivepath} ...')
    with requests.get(url, stream=True) as r:
        with open(archivepath, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    print(f'Extracting {archivepath} to {save_path} ...')
    save_path.mkdir(exist_ok=True)
    call(["tar", f"-zxf{'v' if verbose else ''}", archivepath, "-C", save_path])



DATASET_MAPPING = {
    'maestro': muspy.MAESTRODatasetV3,
    'midicaps': partial(download_from_url, 'https://huggingface.co/datasets/amaai-lab/MidiCaps/resolve/main/midicaps.tar.gz?download=true')
}


@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def process_data(config: DictConfig):
    save_path = Path(config.data.path) / config.data.dataset_name
    dataset = DATASET_MAPPING[config.data.dataset_name.lower()](
        save_path, download_and_extract=True)
    # dataset.convert()




if __name__ == "__main__":
    process_data()  # pylint: disable=no-value-for-parameter
