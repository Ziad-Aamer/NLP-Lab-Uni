from transformers import AutoModelForSequenceClassification, BertConfig
from config import BIOBERT_PATH

def get_model(num_labels, use_dropout=False):
    if use_dropout:
        config = BertConfig.from_pretrained(BIOBERT_PATH)
        config.hidden_dropout_prob = 0.3
        config.attention_probs_dropout_prob = 0.3
        config.num_labels = num_labels
        return AutoModelForSequenceClassification.from_pretrained(BIOBERT_PATH, config=config)
    else:
        return AutoModelForSequenceClassification.from_pretrained(BIOBERT_PATH, num_labels=num_labels)