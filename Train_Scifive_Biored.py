from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import load_from_disk
import numpy as np
import evaluate
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

MODEL_NAME = "razent/SciFive-large-Pubmed"
DATASET_PATH = "biored_tokenized"
OUTPUT_DIR = "scifive_biored_output"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
dataset = load_from_disk(DATASET_PATH)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Optional: basic ROUGE metric for string similarity
eval_metric = evaluate.load("rouge")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return eval_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",  # Changed from evaluation_strategy
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=5,
    predict_with_generate=True,
    logging_dir=f"{OUTPUT_DIR}/logs",
    fp16=False  # Change to True if using GPU with mixed precision
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["dev"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model(f"{OUTPUT_DIR}/final_model")