import os
from textClassifier.logging import logger
from datasets import load_dataset
from pathlib import Path
from textClassifier.utils.common import get_size
from textClassifier.entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self):
        if not os.path.exists(self.config.local_data_file):
            dataset = load_dataset("imdb")
            dataset.save_to_disk(self.config.local_data_file)
            logger.info(f"data download!")
        else:
            logger.info(f"data already exists of size: {get_size(Path(self.config.local_data_file))}")
