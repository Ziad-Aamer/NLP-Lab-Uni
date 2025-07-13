import torch
from sklearn.metrics import classification_report
from config import DEVICE
from tqdm import tqdm
import os
from config import REPORT_DIR

def evaluate_model(model, dataloader, label_list, split_name="eval"):
    model.eval()
    all_preds = []
    all_labels = []

    print("\nEvaluating...")
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    label_ids = list(range(len(label_list)))

    report_dict = classification_report(
        y_true=all_labels,
        y_pred=all_preds,
        labels=label_ids,
        target_names=label_list,
        digits=4,
        zero_division=0,
        output_dict=True
    )

    report_text = classification_report(
        y_true=all_labels,
        y_pred=all_preds,
        labels=label_ids,
        target_names=label_list,
        digits=4,
        zero_division=0
    )

    # Extract macro scores
    macro_prec = report_dict["macro avg"]["precision"]
    macro_recall = report_dict["macro avg"]["recall"]
    macro_f1 = report_dict["macro avg"]["f1-score"]
    accuracy = report_dict["accuracy"]

    # Print full report
    print("\nEvaluation Report:")
    print(report_text)

    # Print summary line
    print(f"\nSummary (macro avg): Accuracy={accuracy:.4f}, Precision={macro_prec:.4f}, Recall={macro_recall:.4f}, F1={macro_f1:.4f}")

    # Save report to file
    output_path = os.path.join(REPORT_DIR, f"{split_name}_report.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
        f.write("\n")
        f.write(f"Summary (macro avg): Accuracy={accuracy:.4f}, Precision={macro_prec:.4f}, Recall={macro_recall:.4f}, F1={macro_f1:.4f}\n")

    print(f"Report saved to {output_path}")
