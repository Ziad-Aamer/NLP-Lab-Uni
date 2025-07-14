import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
import re
import os

model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

def get_prediction(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = model.generate(**inputs, max_new_tokens=2)
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return pred.strip()

def extract_valid_option(pred, valid_options):
    pred = pred.strip().upper()
    if len(pred) > 0 and pred[0] in valid_options:
        return pred[0]
    match = re.search(r'([A-Z])', pred)
    if match and match.group(1) in valid_options:
        return match.group(1)
    return "INVALID"

def evaluate_split(file_path, valid_options, save_results_path):
    df = pd.read_csv(file_path)
    results = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {os.path.basename(file_path)}"):
        prompt = row["qa4re_prompt"]
        gold = row["gold_option_letter"]
        pred = get_prediction(prompt)
        results.append({"gold": gold, "pred": pred})
    df["pred"] = [r["pred"] for r in results]
    df["pred"] = df["pred"].apply(lambda x: extract_valid_option(x, valid_options))
    df["gold"] = [r["gold"] for r in results]
    df["correct"] = df["gold"] == df["pred"]

    # print(df[df["pred"] == "INVALID"][["qa4re_prompt", "pred"]].head())

    # Count invalid predictions
    num_invalid = (df["pred"] == "INVALID").sum()
    percent_invalid = num_invalid / len(df) * 100 if len(df) > 0 else 0

    y_true = df["gold"]
    y_pred = df["pred"]

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    # Save results
    os.makedirs(os.path.dirname(save_results_path), exist_ok=True)
    df.to_csv(save_results_path, index=False)
    return {
        "size": len(df),
        "accuracy": acc,
        "f1": f1,
        "num_invalid": num_invalid,
        "percent_invalid": percent_invalid,
        "df": df
    }


# File paths
dev_file = "prepared_prompt_files/qa4re_prompts_with_gold_DEV.csv"
train_file = "prepared_prompt_files/qa4re_prompts_with_gold_TRAIN.csv"

valid_options = set(["A", "B", "C", "D", "E", "F", "G", "H"])

# Evaluate both splits
dev_metrics = evaluate_split(dev_file, valid_options, "results/qa4re_predictions_DEV.csv")
train_metrics = evaluate_split(train_file, valid_options, "results/qa4re_predictions_TRAIN.csv")

# Print summary
print("\n=== DEV Split ===")
print(f"Size: {dev_metrics['size']}")
print(f"Accuracy: {dev_metrics['accuracy']:.2%}")
print(f"F1 (macro): {dev_metrics['f1']:.4f}")
print(f"Invalid predictions: {dev_metrics['num_invalid']} ({dev_metrics['percent_invalid']:.2f}%)")

print("\n=== TRAIN Split ===")
print(f"Size: {train_metrics['size']}")
print(f"Accuracy: {train_metrics['accuracy']:.2%}")
print(f"F1 (macro): {train_metrics['f1']:.4f}")
print(f"Invalid predictions: {train_metrics['num_invalid']} ({train_metrics['percent_invalid']:.2f}%)")

# Write summary results to a file
summary_path = "results/summary_metrics.txt"
with open(summary_path, "w") as f:
    f.write("=== DEV Split ===\n")
    f.write(f"Size: {dev_metrics['size']}\n")
    f.write(f"Accuracy: {dev_metrics['accuracy']:.2%}\n")
    f.write(f"F1 (macro): {dev_metrics['f1']:.4f}\n")
    f.write(f"Invalid predictions: {dev_metrics['num_invalid']} ({dev_metrics['percent_invalid']:.2f}%)\n\n")
    f.write("=== TRAIN Split ===\n")
    f.write(f"Size: {train_metrics['size']}\n")
    f.write(f"Accuracy: {train_metrics['accuracy']:.2%}\n")
    f.write(f"F1 (macro): {train_metrics['f1']:.4f}\n")
    f.write(f"Invalid predictions: {train_metrics['num_invalid']} ({train_metrics['percent_invalid']:.2f}%)\n")

# Bar plot
labels = ["Train", "Dev"]
accs = [train_metrics['accuracy'], dev_metrics['accuracy']]
f1s = [train_metrics['f1'], dev_metrics['f1']]

x = np.arange(len(labels))  # label locations
width = 0.35  # width of the bars

fig, ax = plt.subplots(figsize=(7, 5))

bars1 = ax.bar(x - width/2, accs, width, label='Accuracy')
bars2 = ax.bar(x + width/2, f1s, width, label='F1-score (macro)')

ax.set_ylabel('Score')
ax.set_title('QA4RE: Accuracy and F1-score Comparison')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1)
ax.legend()

# Annotate bars with value
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig("results/train_dev_accuracy_f1_comparison.png")
plt.show()
