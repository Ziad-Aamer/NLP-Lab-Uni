from transformers import AutoModelForSequenceClassification
from config import BIOBERT_PATH

def get_model(num_labels):
    return AutoModelForSequenceClassification.from_pretrained(
        BIOBERT_PATH,
        num_labels=num_labels
    )