import csv
import random
import numpy as np
import torch

def extract_labels_from_tsv(tsv_path):
    label_set = set()
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) != 4:
                continue
            label = row[3].strip()
            label_set.add(label)

    label_list = sorted(label_set)  # consistent order
    label_to_id = {label: idx for idx, label in enumerate(label_list)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    return label_list, label_to_id, id_to_label

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

