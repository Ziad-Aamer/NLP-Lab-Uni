# predict_scifive.py
import sys
import csv
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_PATH = "scifive_biored_output/final_model"
INPUT_FILE = "predict_input.txt"
OUTPUT_FILE = "predictions.csv"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
model.eval()

def predict(text: str, max_length: int = 128):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=4,
            repetition_penalty=2.5,
            early_stopping=True,
            no_repeat_ngram_size=4,
            do_sample=False  # <== important: disables randomness
    )    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    with open(INPUT_FILE, "r") as infile:
        inputs = [line.strip() for line in infile if line.strip()]

    with open(OUTPUT_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Input", "Prediction"])
        for line in inputs:
            output = predict(line)
            writer.writerow([line, output])

    print(f"Predictions written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()