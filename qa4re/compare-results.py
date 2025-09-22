import pandas as pd
import matplotlib.pyplot as plt

# === Step 1: Put results into a DataFrame ===
data = {
    "Model": [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "meta-llama/Llama-3.2-3B-Instruct"
        # "meta-llama/Llama-3.1-8B-Instruct",
        # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        # "openai/gpt-4o",
    ],
    "Accuracy": [21, 23],
    "F1": [11, 13],
    "Dataset": ["Test with Few-shots"] * 2,
}

df = pd.DataFrame(data)
df = df.sort_values(by="F1", ascending=False)

# === Step 2: Find top 2 per metric ===
def get_top_models(df, metric):
    return df.sort_values(metric, ascending=False).head(2)["Model"].tolist()

# === Step 3: Plotting ===
metrics = ["Accuracy", "F1"]
colors = {"default": "skyblue", "top": "orange", "second": "green"}

fig, axes = plt.subplots(1, len(metrics), figsize=(12, 5), sharey=True)

# find max across all metrics for shared scale
xmax = df[metrics].values.max() * 1.1  # add 10% headroom

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    values = df[metric]
    models = df["Model"]

    top_models = get_top_models(df, metric)

    bar_colors = []
    for model in models:
        if model == top_models[0]:
            bar_colors.append(colors["top"])
        elif model == top_models[1]:
            bar_colors.append(colors["second"])
        else:
            bar_colors.append(colors["default"])

    bars = ax.barh(models, values, color=bar_colors)
    ax.set_title(f"{metric} (%)")
    ax.set_xlabel("Score")
    ax.set_xlim(0, xmax)   # enforce same scale

    # Annotate bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                f"{width:.1f}", va="center")

dataset_name = df["Dataset"].iloc[0]
plt.suptitle(f"Model Comparison on {dataset_name} Dataset: Accuracy vs F1 (macro)", fontsize=14)

plt.tight_layout()
plt.savefig(f"model_comparison_{dataset_name.lower()}.png", dpi=300, bbox_inches="tight")
plt.show()
