import torch
from transformers import AdamW
from tqdm import tqdm
from config import DEVICE, EPOCHS, LR, MODEL_DIR, REPORT_DIR
from evaluate import evaluate_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

def train_model(model, train_loader, dev_loader, label_list):
    model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()

    log_file = os.path.join(REPORT_DIR, "training_report.txt")
    with open(log_file, "w", encoding="utf-8") as log_f:

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            all_preds = []
            all_labels = []

            log_f.write(f"\nEpoch {epoch+1}/{EPOCHS}\n")
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
            for step, batch in progress_bar:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                loss = loss_fn(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                avg_loss = total_loss / (step + 1)
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{avg_loss:.4f}"
                })

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                if (step + 1) % 10 == 0 or (step + 1) == len(train_loader):
                    avg_loss = total_loss / (step + 1)
                    acc = accuracy_score(all_labels, all_preds)
                    log_line = (
                        f"Step {step+1}/{len(train_loader)} - "
                        f"Batch Loss: {loss.item():.4f} - "
                        f"Running Avg Loss: {avg_loss:.4f} - "
                        f"Accuracy: {acc:.4f}"
                    )
                    log_f.write(log_line + "\n")

            # Epoch-level metrics
            epoch_loss = total_loss / len(train_loader)
            acc = accuracy_score(all_labels, all_preds)
            prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
            rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
            f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

            print(f"Train loss: {epoch_loss:.4f}")
            print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
            log_f.write(f"Epoch {epoch+1} Summary - Loss: {epoch_loss:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}\n\n")

            evaluate_model(model, dev_loader, label_list, split_name=f"dev_epoch{epoch+1}")

            # Save model
            save_path = os.path.join(MODEL_DIR, f"biobert_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Saved model checkpoint to {save_path}")
