import pandas as pd
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import re
import os
import litellm


# File paths
dev_file = "prepared_prompt_files/qa4re_prompts_with_gold_DEV.csv"
train_file = "prepared_prompt_files/qa4re_prompts_with_gold_TRAIN.csv"
mini_file = "prepared_prompt_files/qa4re_prompts_with_gold_MINI.csv"

valid_options = set(["A", "B", "C", "D", "E", "F", "G", "H"])


def get_prediction(prompt, model_name):
    response = litellm.completion(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
        api_key="",
        api_base="http://131.220.150.238:8080"  # <- force the proxy endpoint
    )
    return response["choices"][0]["message"]["content"].strip()


def extract_valid_option(pred, valid_options):
    pred = pred.strip().upper()
    if len(pred) > 0 and pred[0] in valid_options:
        return pred[0]
    match = re.search(r'([A-Z])', pred)
    if match and match.group(1) in valid_options:
        return match.group(1)
    
    return "_INVALID_" + pred

def evaluate_split(file_path, valid_options, save_results_path, model_name):
    df = pd.read_csv(file_path)
    results = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {os.path.basename(file_path)} with {model_name}"):
        prompt = row["qa4re_prompt"]
        gold = row["gold_option_letter"]
        pred = get_prediction(prompt, model_name)
        results.append({"gold": gold, "pred": pred})
        
    df["pred"] = [extract_valid_option(r["pred"], valid_options) for r in results]
    df["gold"] = [r["gold"] for r in results]
    df["correct"] = df["gold"] == df["pred"]

    num_invalid = df["pred"].apply(lambda x: x.startswith("_INVALID_")).sum()
    percent_invalid = num_invalid / len(df) * 100 if len(df) > 0 else 0

    y_true = df["gold"]
    y_pred = df["pred"]

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

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

def print_evaluation(metrics_dict):
    for model_name, splits in metrics_dict.items():
        print(f"\n=== Results for {model_name} ===")
        for split_name, m in splits.items():
            print(f"\n--- {split_name.upper()} Split ---")
            print(f"Size: {m['size']}")
            print(f"Accuracy: {m['accuracy']:.2%}")
            print(f"F1 (macro): {m['f1']:.4f}")
            print(f"Invalid predictions: {m['num_invalid']} ({m['percent_invalid']:.2f}%)")
            # Write summary results to a file
            summary_path = f"results/summary_metrics.txt"
            with open(summary_path, "w") as f:
                for split_name, m in splits.items():
                    f.write(f"=== {split_name.upper()} Split ({model_name}) ===\n")
                    f.write(f"Size: {metrics_dict[model_name][split_name]['size']}\n")
                    f.write(f"Accuracy: {metrics_dict[model_name][split_name]['accuracy']:.2%}\n")
                    f.write(f"F1 (macro): {metrics_dict[model_name][split_name]['f1']:.4f}\n")
                    f.write(f"Invalid predictions: {metrics_dict[model_name][split_name]['num_invalid']} ({metrics_dict[model_name][split_name]['percent_invalid']:.2f}%)\n\n")

def plot_results(metrics_dict):

    for model_name, splits in metrics_dict.items():
        accs = [splits[split]["accuracy"] for split in splits]
        f1s = [splits[split]["f1"] for split in splits]
        labels = [split.capitalize() for split in splits]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(7, 5))

        bars1 = ax.bar(x - width/2, accs, width, label='Accuracy')
        bars2 = ax.bar(x + width/2, f1s, width, label='F1-score (macro)')

        ax.set_ylabel('Score')
        ax.set_title(f'QA4RE: Accuracy and F1-score ({model_name})')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.legend()

        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

        os.makedirs("results", exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"results/{model_name.replace('/', '_')}_train_dev_accuracy_f1.png")
        plt.show()

if __name__ == "__main__":
    metrics_dict = {}
    # models = ["openai/gpt-4o", "openai/gpt-4o-mini"]
    splits = {
        "dev": dev_file,
        "train": train_file
        # "mini": mini_file
    }
    
    # Run both models
    for model_name in ["openai/gpt-4"]:
        metrics_dict[model_name] = {}
        for split_name, file_path in splits.items():
            metrics = evaluate_split(
                file_path,
                valid_options,
                f"results/{model_name.replace('/', '_')}_{split_name.upper()}.csv",
                model_name
            )
            metrics_dict[model_name][split_name] = metrics

    print_evaluation(metrics_dict)
    plot_results(metrics_dict)
