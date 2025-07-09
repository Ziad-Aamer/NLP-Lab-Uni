from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from datasets import load_from_disk

# Load SciFive tokenizer
MODEL_NAME = "razent/SciFive-large-Pubmed"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load preprocessed dataset
DATASET_PATH = "biored_scifive_text2text"
dataset = load_from_disk(DATASET_PATH)

# Tokenization function
def tokenize_function(example):
    model_inputs = tokenizer(
        example["input"],
        max_length=512,
        padding="max_length",
        truncation=True
    )
    labels = tokenizer(
        example["output"],
        max_length=128,
        padding="max_length",
        truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply to all splits
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Data collator for Trainer
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=MODEL_NAME)

# Optionally save
tokenized_datasets.save_to_disk("biored_tokenized")