import os
import re
import matplotlib.pyplot as plt

from config import REPORT_DIR, OUTPUT_DIR

def read_epoch_metrics():
    log_path = os.path.join(REPORT_DIR, "training_report.txt")
    epoch_data = {
        "epoch": [],
        "train_acc": [],
        "train_f1": [],
        "dev_acc": [],
        "dev_f1": [],
    }

    with open(log_path, encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.startswith("Epoch") and "Summary" in line:
            parts = line.split("|")
            epoch_num = int(re.findall(r'\d+', parts[0])[0])
            train_acc = float(parts[1].split(":")[1].strip())
            train_f1 = float(parts[4].split(":")[1].strip())

            epoch_data["epoch"].append(epoch_num)
            epoch_data["train_acc"].append(train_acc)
            epoch_data["train_f1"].append(train_f1)

    # Now load each dev report
    for epoch in epoch_data["epoch"]:
        report_file = os.path.join(REPORT_DIR, f"dev_epoch{epoch}_report.txt")
        if os.path.exists(report_file):
            with open(report_file, encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                if line.startswith("Summary"):
                    acc = float(re.findall(r"Accuracy=([0-9.]+)", line)[0])
                    f1 = float(re.findall(r"F1=([0-9.]+)", line)[0])
                    epoch_data["dev_acc"].append(acc)
                    epoch_data["dev_f1"].append(f1)
                    break
        else:
            epoch_data["dev_acc"].append(None)
            epoch_data["dev_f1"].append(None)

    return epoch_data


def read_test_metrics():
    path = os.path.join(REPORT_DIR, "test_report.txt")
    if not os.path.exists(path):
        return None, None
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith("Summary"):
            acc = float(re.findall(r"Accuracy=([0-9.]+)", line)[0])
            f1 = float(re.findall(r"F1=([0-9.]+)", line)[0])
            return acc, f1
    return None, None


def plot_metric_curve(metric_name, train_vals, dev_vals, test_val=None):
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)
    epochs = list(range(1, len(train_vals) + 1))

    plt.figure()
    plt.plot(epochs, train_vals, label="Train")
    plt.plot(epochs, dev_vals, label="Dev")

    # Highlight best dev epoch
    if dev_vals:
        best_epoch = int(np.argmax(dev_vals)) + 1
        plt.axvline(x=best_epoch, linestyle="dotted", color="gray", label=f"Best Dev Epoch = {best_epoch}")

    # Horizontal test line
    if test_val is not None:
        plt.axhline(y=test_val, xmin=0, xmax=len(epochs), linestyle="dashed", color="red")
        plt.text(
            x=epochs[-1], y=test_val,
            s=f"Test {metric_name} = {test_val:.4f}",
            color="red", fontsize=10, ha="right", va="bottom"
        )

    plt.title(f"{metric_name} over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plots", f"{metric_name.lower()}_curve.png"))
    plt.close()

def generate_all_plots():
    data = read_epoch_metrics()
    test_acc, test_f1 = read_test_metrics()

    plot_metric_curve("Accuracy", data["train_acc"], data["dev_acc"], test_acc)
    plot_metric_curve("F1", data["train_f1"], data["dev_f1"], test_f1)