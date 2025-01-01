from textClassifier.config.configuration import ConfigurationManager
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import pipeline
import os


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_predict_config()

    def predict(self, text):
        model = BertForSequenceClassification.from_pretrained(os.path.join(self.config.model_path, "model"))
        tokenizer = BertTokenizer.from_pretrained(os.path.join(self.config.tokenizer_path, "tokenizer"))

        sentiment_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

        print("Text:")
        print(text)

        output = sentiment_pipeline(text)

        for item in output:
            if item['label'] == 'LABEL_1':
                item['label'] = 'positive'
            elif item['label'] == 'LABEL_0':
                item['label'] = 'negative'

        print("\nModel Prediction:")
        print(output)

        return output
