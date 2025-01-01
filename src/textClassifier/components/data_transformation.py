import os
from textClassifier.logging import logger
from transformers import BertTokenizer
from datasets import load_dataset, load_from_disk
from textClassifier.entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config.tokenizer_name)

    def tokenize_data(self, example):
        return self.tokenizer(example['text'], padding='max_length', truncation=True, max_length=512)

    def convert(self):
        dataset = load_from_disk(self.config.data_path)
        encoded_dataset = dataset.map(self.tokenize_data, batched=True)
        encoded_dataset.save_to_disk(os.path.join(self.config.root_dir, "imdb"))