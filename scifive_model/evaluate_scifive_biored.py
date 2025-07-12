# evaluate_scifive_biored.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_from_disk
import evaluate
from tqdm import tqdm
import torch

# Paths
MODEL_DIR = "scifive_biored_output/final_model"
DATASET_DIR = "biored_scifive_text2text"  # previously saved with save_to_disk()

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
model.eval()

# Load test set
dataset = load_from_disk(DATASET_DIR)
test_data = dataset["test"]

# Load metric
rouge = evaluate.load("rouge")

# Generate predictions
preds = []
refs = []

for sample in tqdm(test_data, desc="Evaluating"):
    input_ids = tokenizer(sample["input"], return_tensors="pt", truncation=True).input_ids
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, max_length=128)
    pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    preds.append(pred_text)
    refs.append(sample["output"])

# Compute and print ROUGE
results = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
print("\n=== Evaluation Results ===")
for k, v in results.items():
    print(f"{k}: {v:.4f}")