import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import re
import os

from huggingface_hub import login
login("")  # <- put your HF token here if needed


# -----------------
# Model setup
# -----------------
# google/flan-t5-xl
# google/flan-t5-large
# google/flan-t5-base
# google/flan-t5-small
# mistralai/Mistral-7B-Instruct-v0.3
# meta-llama/Llama-3.2-3B-Instruct
# meta-llama/Llama-3.1-8B-Instruct
# deepseek-ai/DeepSeek-R1-Distill-Llama-8B
# openai/gpt-4o
# google/gemma-3-4b-it


model_name = "mistralai/Mistral-7B-Instruct-v0.3"

datasets = {
    # "prepared_prompt_files/qa4re_prompts_with_gold_MINI.csv": "MINI"
    # "prepared_prompt_files/qa4re_prompts_with_gold_DEV.csv": "DEV",
    # "prepared_prompt_files/qa4re_prompts_with_gold_TRAIN.csv": "TRAIN"
    "prepared_prompt_files/qa4re_prompts_with_gold_TEST.csv": "TEST"
    # Add more datasets here
}


def load_model(model_name, device="cuda"):
    """Load model & tokenizer and detect type (seq2seq vs causal)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)

    try:
        # Try seq2seq first
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        model_type = "seq2seq"
    except Exception:
        # Fall back to causal LM
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto", use_auth_token=True)
        model_type = "causal"

    return tokenizer, model, model_type


def get_prediction(prompt):
    """Generate prediction for either seq2seq or causal LM."""
    if model_type == "seq2seq":
        # T5, BART, etc.
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        outputs = model.generate(**inputs, max_new_tokens=2)
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)

    elif model_type == "causal":
        # Mistral, LLaMA, GPT-style
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        outputs = model.generate(**inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id)
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # For causal LM, remove the original prompt from output
        if pred.startswith(prompt):
            pred = pred[len(prompt):].strip()

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return pred.strip()


def extract_valid_option(pred):
    valid_options = set(["A", "B", "C", "D", "E", "F", "G", "H"])

    pred = pred.strip().upper()
    if len(pred) > 0 and pred[0] in valid_options:
        return pred[0]
    match = re.search(r'([A-Z])', pred)
    if match and match.group(1) in valid_options:
        return match.group(1)
    return "_INVALID_" + pred

def evaluate_split(file_path, dataset_name, save_results_path):
    df = pd.read_csv(file_path)
    results = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {dataset_name}"):
        prompt = row["qa4re_prompt"]
        gold = row["gold_option_letter"]
        pred = get_prediction(prompt)
        results.append({"gold": gold, "pred": pred})

    df["pred"] = [r["pred"] for r in results]
    df["pred"] = df["pred"].apply(lambda x: extract_valid_option(x))
    df["gold"] = [r["gold"] for r in results]
    df["correct"] = df["gold"] == df["pred"]

    num_invalid = df["pred"].apply(lambda x: str(x).startswith("_INVALID_")).sum()
    percent_invalid = num_invalid / len(df) * 100 if len(df) > 0 else 0

    y_true, y_pred = df["gold"], df["pred"]
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    os.makedirs(os.path.dirname(save_results_path), exist_ok=True)
    df.to_csv(save_results_path, index=False)

    return {
        "dataset": dataset_name,
        "size": len(df),
        "accuracy": acc,
        "f1": f1,
        "num_invalid": num_invalid,
        "percent_invalid": percent_invalid
    }

# -----------------
# Multiple datasets (dict: filepath -> dataset name)
# -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer, model, model_type = load_model(model_name, device=device)

all_metrics = []

for file_path, dataset_name in datasets.items():
    save_path = f"results/{dataset_name}_predictions.csv"
    metrics = evaluate_split(file_path, dataset_name, save_path)
    all_metrics.append(metrics)

# -----------------
# Save summary
# -----------------
summary_path = "results/summary_metrics.txt"
with open(summary_path, "w") as f:
    for m in all_metrics:
        f.write(f"=== {m['dataset']} ===\n")
        f.write(f"Size: {m['size']}\n")
        f.write(f"Accuracy: {m['accuracy']:.2%}\n")
        f.write(f"F1 (macro): {m['f1']:.4f}\n")
        f.write(f"Invalid predictions: {m['num_invalid']} ({m['percent_invalid']:.2f}%)\n\n")

# -----------------
# Plot results
# -----------------
labels = [m["dataset"] for m in all_metrics]
accs = [m["accuracy"] for m in all_metrics]
f1s = [m["f1"] for m in all_metrics]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, accs, width, label="Accuracy")
bars2 = ax.bar(x + width/2, f1s, width, label="F1-score (macro)")

ax.set_ylabel("Score")
ax.set_title("Accuracy and F1-score Comparison across datasets")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15)
ax.set_ylim(0, 1)
ax.legend()

for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(f"results/{model_name.replace('/', '_')}_datasets_accuracy_f1.png")
plt.show()
