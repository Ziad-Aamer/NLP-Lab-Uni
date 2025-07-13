from preprocess import convert_bioc_to_tsv
from dataset import load_dataloaders
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, BATCH_SIZE, MAX_LEN, DEVICE, REPORT_DIR, MODEL_DIR
from model import get_model
from train import train_model
from evaluate import evaluate_model
from utils import set_seed
import argparse
import os

def main():

    set_seed(42)  # Set random seed for reproducibility

    parser = argparse.ArgumentParser()
    parser.add_argument('--mini', action='store_true', help='Use mini-biored as the dataset')
    parser.add_argument('--gen-result-plots', action='store_true', help='Generate evaluation metric plots and exit')
    args = parser.parse_args()

    # Early return if plotting only
    if args.gen_result_plots:
        from plot import generate_all_plots
        generate_all_plots()
        print("Plots saved to outputs/plots/")
        return

    if args.mini:
        print("Using mini-biored.JSON for quick testing")
        mini_path = "../qa4re/mini-biored.JSON"
        from preprocess import convert_single_bioc_to_tsv

        mini_out = os.path.join(PROCESSED_DATA_DIR, "mini")
        os.makedirs(mini_out, exist_ok=True)

        for split in ["train", "dev", "test"]:
            convert_single_bioc_to_tsv(mini_path, os.path.join(mini_out, f"{split}.tsv"))
        active_data_dir = mini_out

    else:
        print("Running preprocessing...")
        convert_bioc_to_tsv(
            input_dir=RAW_DATA_DIR,
            output_dir=PROCESSED_DATA_DIR,
            splits=["Train", "Dev", "Test"]
        )
        active_data_dir = PROCESSED_DATA_DIR

    print("Loading datasets...")
    train_loader, dev_loader, test_loader, label_list, label_to_id, id_to_label = load_dataloaders(
        data_dir=active_data_dir,
        batch_size=BATCH_SIZE,
        max_len=MAX_LEN
    )

    print("Labels:", label_list)
    print("Number of training samples:", len(train_loader.dataset))
    print("Number of dev samples:", len(dev_loader.dataset))
    print("Number of test samples:", len(test_loader.dataset))

    # Dynamically compute num_labels from label list
    num_labels = len(label_list)

    # Instantiate the model
    print("Instantiating model...")
    model = get_model(num_labels).to(DEVICE)

    # Start training
    train_model(model, train_loader, dev_loader, label_list)

    # Evaluate on the test set
    print("\nEvaluating on test set:")
    evaluate_model(model, test_loader, label_list, split_name="test")

    print(f"All model checkpoints saved to: {MODEL_DIR}")
    print(f"Reports and logs saved to: {REPORT_DIR}")

if __name__ == "__main__":
    main()
