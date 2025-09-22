# REBEL fine-tuning on BioRED

This directory contains scripts to fine-tune the `Babelscape/rebel-large` model on the BioRED relation extraction benchmark and to evaluate the model with per-class precision, recall, and F1 scores.

## Workflow

1. **Prepare the data** – convert the BioC JSON files in `BioRED/dataset/` into REBEL-style input/target pairs:
   ```bash
   python REBEL/scripts/prepare_biored_rebel.py
   ```
   The script writes JSONL files to `REBEL/data/`.

2. **Train and evaluate** – run the seq2seq training script:
   ```bash
   python REBEL/scripts/train_rebel.py \
     --do-train --do-eval --do-predict \
     --model-name-or-path Babelscape/rebel-large \
     --output-dir REBEL/models/rebel-biored
   ```
   Additional hyperparameters (batch sizes, epochs, learning rate, etc.) can be supplied via CLI arguments. The script logs metrics to the console and saves `train_results.json`, `eval_results.json`, and `test_results.json` inside the output directory. Each JSON file includes per-class `precision_*`, `recall_*`, and `f1_*` keys.

3. **Submit to SLURM** – use the batch script to run on the RWTH cluster:
   ```bash
   sbatch REBEL/run_rebel.sh
   ```
   Logs are stored under `REBEL/logs/` and the fine-tuned model plus metrics under `REBEL/models/rebel-biored/`.

## Requirements

Install the Python dependencies listed in `REBEL/requirements.txt`. The SLURM script does this automatically with the cluster modules already used elsewhere in the project.

## Outputs

- `REBEL/data/biored_rebel_{split}.jsonl`: processed dataset splits.
- `REBEL/models/<run-name>/`: Hugging Face trainer outputs, checkpoints, and metric JSON files.
- `REBEL/logs/`: SLURM job stdout/err files.

Adjust hyperparameters in `REBEL/run_rebel.sh` or supply custom arguments when calling `train_rebel.py` directly.
