import torch
import os

# ====== PATHS ======
BIOBERT_PATH = "dmis-lab/biobert-base-cased-v1.1"
RAW_DATA_DIR = "../BioRED/dataset/"
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
EPOCHS = 10
LR = 1e-5
# LR = 2e-5
# LR = 5e-6
WARMUP_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 5  # Number of epochs to wait for improvement
EARLY_STOPPING_DELTA = 1e-4  # Minimum change to qualify as improvement

# ====== DEVICE ======
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
