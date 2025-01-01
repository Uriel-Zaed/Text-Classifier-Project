from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_from_disk
from textClassifier.entity import ModelTrainerConfig
import torch
import os


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def compute_metrics(self, pred):
        logits, labels = pred
        predictions = torch.argmax(torch.tensor(logits), dim=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        acc = accuracy_score(labels, predictions)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = BertTokenizer.from_pretrained(self.config.model_name)
        model = BertForSequenceClassification.from_pretrained(self.config.model_name).to(device)

        # loading data
        dataset = load_from_disk(self.config.data_path)

        training_args = TrainingArguments(
            output_dir=str(self.config.output_dir),
            evaluation_strategy=self.config.evaluation_strategy,
            save_strategy=self.config.save_strategy,
            logging_dir=str(self.config.logging_dir),
            logging_steps=self.config.logging_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            num_train_epochs=self.config.num_train_epochs,
            load_best_model_at_end=self.config.load_best_model_at_end,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            compute_metrics=self.compute_metrics
        )

        trainer.train()

        # Save model
        model.save_pretrained(os.path.join(self.config.root_dir, "model"))
        # Save tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "tokenizer"))
