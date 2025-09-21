from preprocess import convert_bioc_to_tsv
from dataset import load_dataloaders
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, BATCH_SIZE, MAX_LEN, DEVICE, REPORT_DIR, MODEL_DIR
from model import get_model
from train import train_model
from evaluate import evaluate_model
from utils import set_seed
import argparse
import os
import torch

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--mini', action='store_true', help='Use mini-biored as the dataset')
    parser.add_argument('--gen-result-plots', action='store_true', help='Generate evaluation metric plots and exit')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with verbose logging')
    parser.add_argument('--early-stopping', action='store_true', help='Enable early stopping during training')
    parser.add_argument('--save-model', action='store_true', help='Save the trained model after training')
    parser.add_argument('--weighted-loss', action='store_true', help='Use class-weighted loss')
    parser.add_argument('--dropout', action='store_true', help='Enable dropout on classifier head')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay for optimizer')
    parser.add_argument('--show-confusion', action='store_true', help='Show confusion matrix')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument("--use-best", action="store_true", help="Load best model (by dev F1) for test evaluation")
    parser.add_argument("--max-seq-len", type=int, default=MAX_LEN, help="Maximum input sequence length for tokenizer and model")
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Apply label smoothing loss (value between 0 and 1)')
    parser.add_argument('--focal-loss', action='store_true', help='Use focal loss instead of cross entropy')

    args = parser.parse_args()

    set_seed(args.seed)  # Set random seed for reproducibility

    print(f"[DEBUG] : {args.debug}")

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
        max_len=args.max_seq_len
    )

    if args.debug:
        print("[DEBUG] Labels:", label_list)
        print("[DEBUG] Number of training samples:", len(train_loader.dataset))
        print("[DEBUG] Number of dev samples:", len(dev_loader.dataset))
        print("[DEBUG] Number of test samples:", len(test_loader.dataset))

    # Instantiate the model
    print("Instantiating model...")
    model = get_model(len(label_list), use_dropout=args.dropout).to(DEVICE)

    # Start training
    train_model(
        model,
        train_loader,
        dev_loader,
        label_list,
        debug=args.debug,
        epoch_override=not args.early_stopping,
        save_model=args.save_model,
        use_weighted_loss=args.weighted_loss,
        weight_decay=args.weight_decay,
        show_confusion=args.show_confusion,
        label_smoothing=args.label_smoothing,
        use_focal_loss=args.focal_loss
    )

    if args.use_best:
        best_path = os.path.join(MODEL_DIR, "best_model.pt")
        if os.path.exists(best_path):
            print(f"[INFO] Loading best model from {best_path}")
            model.load_state_dict(torch.load(best_path, map_location=DEVICE))
            model.to(DEVICE)
        else:
            print(f"[WARN] Best model not found at {best_path}, using current model instead.")

    # Evaluate on the test set
    print("\nEvaluating on test set:")
    evaluate_model(model, 
        test_loader,
        label_list,
        split_name="test",
        debug=args.debug,
        show_confusion=args.show_confusion
    )

    if args.save_model:
        print(f"All model checkpoints saved to: {MODEL_DIR}")
    print(f"Reports and logs saved to: {REPORT_DIR}")

    # Generate plots
    from plot import generate_all_plots
    generate_all_plots()
    print("Plots saved to outputs/plots/")

if __name__ == "__main__":
    main()
