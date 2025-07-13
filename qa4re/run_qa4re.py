import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
file_path = "prepared_prompt_files/qa4re_prompts_with_gold_TRAIN.csv"  # adjust path as needed

df = pd.read_csv(file_path)

def get_prediction(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = model.generate(**inputs, max_new_tokens=2)
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return pred.strip()

results = []
for i, row in tqdm(df.iterrows(), total=len(df)):
    prompt = row["qa4re_prompt"]
    gold = row["gold_option_letter"]  # e.g. "A"
    pred = get_prediction(prompt)
    results.append({"gold": gold, "pred": pred})

df["pred"] = [r["pred"] for r in results]
df["gold"] = [r["gold"] for r in results]
df["correct"] = df["gold"] == df["pred"]

# Metrics
y_true = df["gold"]
y_pred = df["pred"]

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
print(classification_report(y_true, y_pred, zero_division=0))


print(f"Accuracy: {acc:.2%}")
print(f"F1 Score (weighted): {f1:.4f}")
print(f"Precision (weighted): {prec:.4f}")
print(f"Recall (weighted): {rec:.4f}")

# Optional: show a full classification report (per-class metrics)
print("\nFull classification report:\n")
print(classification_report(y_true, y_pred, zero_division=0))

results_path = "results/qa4re_predictions.csv"
df.to_csv(results_path, index=False)
