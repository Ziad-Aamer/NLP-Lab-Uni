import torch
import os

# ====== PATHS ======
BIOBERT_PATH = "dmis-lab/biobert-base-cased-v1.1"
RAW_DATA_DIR = "../BioRed/dataset/"
PROCESSED_DATA_DIR = "./data/"
OUTPUT_DIR = "./outputs/"
MODEL_DIR = os.path.join(OUTPUT_DIR, "model_checkpoints")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

# Ensure output directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ====== MODEL & TRAINING ======
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5
WARMUP_RATIO = 0.1

# ====== DEVICE ======
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
