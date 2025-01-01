from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    local_data_file: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    ALL_REQUIRED_FILES: list


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    tokenizer_name: Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    output_dir: Path
    evaluation_strategy: str
    save_strategy: str
    logging_dir: Path
    logging_steps: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    num_train_epochs: int
    load_best_model_at_end: bool
    model_name: str
    data_path: Path
    root_dir: Path


@dataclass(frozen=True)
class PredictConfig:
    model_path: Path
    tokenizer_path: Path

