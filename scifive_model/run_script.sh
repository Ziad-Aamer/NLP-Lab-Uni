#!/bin/bash
#SBATCH --job-name=scifive-biored
#SBATCH --output=logs/scifive_%j.out
#SBATCH --error=logs/scifive_%j.err
#SBATCH --account=rwth1809
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --time=0:01:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Load required modules
module purge
module load GCCcore/14.2.0
module load Python/3.13.1
module load CUDA/12.6.3
module load PyTorch/nvcr-25.02-py3

# Install required Python packages in user space
pip install -r requirements.txt


# Enable verbose training logs
export TRANSFORMERS_VERBOSITY=info

# Run training script
python predict_scifive.py