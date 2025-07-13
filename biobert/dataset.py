import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from config import BIOBERT_PATH, MAX_LEN
from utils import extract_labels_from_tsv

SPECIAL_TOKENS = {
    "e1_start": "[E1]",
    "e1_end": "[/E1]",
    "e2_start": "[E2]",
    "e2_end": "[/E2]",
}

def insert_entity_markers(sentence, ent1, ent2):
    # Handle overlapping, repeated, or order-agnostic mentions robustly
    try:
        e1_start = sentence.index(ent1)
        e1_end = e1_start + len(ent1)

        # Replace the first entity with markers
        marked = sentence[:e1_start] + SPECIAL_TOKENS["e1_start"] + ent1 + SPECIAL_TOKENS["e1_end"] + sentence[e1_end:]
        
        # Adjust offset for entity2 (might shift if it occurs after entity1)
        shift = len(SPECIAL_TOKENS["e1_start"]) + len(SPECIAL_TOKENS["e1_end"])
        e2_start = marked.index(ent2, e1_end + shift - len(ent1))
        e2_end = e2_start + len(ent2)

        # Replace the second entity
        marked = marked[:e2_start] + SPECIAL_TOKENS["e2_start"] + ent2 + SPECIAL_TOKENS["e2_end"] + marked[e2_end:]

        return marked
    except ValueError:
        # Fallback: return sentence without markers
        return sentence

class RelationDataset(Dataset):
    def __init__(self, path, label_to_id, tokenizer):
        self.samples = []
        self.label_to_id = label_to_id
        self.tokenizer = tokenizer

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 4:
                    continue
                sentence, ent1, ent2, label = parts
                marked_sentence = insert_entity_markers(sentence, ent1, ent2)
                self.samples.append((marked_sentence, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sentence, label = self.samples[idx]
        encoding = self.tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)      # shape: [max_len]
        attention_mask = encoding["attention_mask"].squeeze(0)

        label_id = self.label_to_id[label]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label_id, dtype=torch.long)
        }

def load_dataloaders(data_dir, batch_size, max_len):
    tokenizer = AutoTokenizer.from_pretrained(BIOBERT_PATH, do_lower_case=False)

    # Extract label set from training data
    train_path = os.path.join(data_dir, "train.tsv")
    label_list, label_to_id, id_to_label = extract_labels_from_tsv(train_path)

    train_set = RelationDataset(train_path, label_to_id, tokenizer)
    dev_set = RelationDataset(os.path.join(data_dir, "dev.tsv"), label_to_id, tokenizer)
    test_set = RelationDataset(os.path.join(data_dir, "test.tsv"), label_to_id, tokenizer)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, dev_loader, test_loader, label_list, label_to_id, id_to_label
