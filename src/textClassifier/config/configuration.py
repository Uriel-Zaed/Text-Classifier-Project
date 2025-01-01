from textClassifier.constants import *
from textClassifier.utils.common import read_yaml, create_directories
from textClassifier.entity import (DataIngestionConfig, DataValidationConfig,
                                   DataTransformationConfig, ModelTrainerConfig, PredictConfig)


class ConfigurationManager:
    def __init__(
            self,
            config_filepath=CONFIG_FILE_PATH,
            params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            local_data_file=config.local_data_file
        )

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            ALL_REQUIRED_FILES=config.ALL_REQUIRED_FILES,
        )

        return data_validation_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            tokenizer_name=config.tokenizer_name
        )

        return data_transformation_config

    #
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.TrainingArguments

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            output_dir=params.output_dir,
            evaluation_strategy=params.evaluation_strategy,
            save_strategy=params.save_strategy,
            logging_dir=params.logging_dir,
            logging_steps=params.logging_steps,
            per_device_train_batch_size=params.per_device_train_batch_size,
            per_device_eval_batch_size=params.per_device_eval_batch_size,
            num_train_epochs=params.num_train_epochs,
            load_best_model_at_end=params.load_best_model_at_end,
            model_name=config.model_name,
            data_path=config.data_path,
            root_dir=config.root_dir
        )

        return model_trainer_config

    def get_predict_config(self) -> PredictConfig:
        config = self.config.predict

        predict_config = PredictConfig(
            model_path=config.model_path,
            tokenizer_path=config.tokenizer_path
        )

        return predict_config
