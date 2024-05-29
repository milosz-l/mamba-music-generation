"""
This is the demo code that uses hydra to access the parameters in under the directory config.

Author: Khuyen Tran
"""

import hydra
from pathlib import Path
import muspy
from omegaconf import DictConfig

DATASET_MAPPING = {'maestro': muspy.MAESTRODatasetV3}


@hydra.main(config_path="../config", config_name="main", version_base="1.2")
def process_data(config: DictConfig):
    save_path = Path(config.data.path) / config.data.dataset_name
    dataset = DATASET_MAPPING[config.data.dataset_name.lower()](
        save_path, download_and_extract=True)
    dataset.convert()


if __name__ == "__main__":
    process_data()
