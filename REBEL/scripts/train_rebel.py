#!/usr/bin/env python
"""Fine-tune the REBEL model on the BioRED dataset and report per-class metrics."""

import argparse
import logging
from typing import Dict, List, Tuple

import numpy as np
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

TRIPLET_TOKEN = "<triplet>"
SUBJ_TOKEN = "<subj>"
OBJ_TOKEN = "<obj>"

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", default="Babelscape/rebel-large")
    parser.add_argument("--train-file", default="REBEL/data/biored_rebel_train.jsonl")
    parser.add_argument("--validation-file", default="REBEL/data/biored_rebel_dev.jsonl")
    parser.add_argument("--test-file", default="REBEL/data/biored_rebel_test.jsonl")
    parser.add_argument("--output-dir", default="REBEL/models/rebel-biored")
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=192)
    parser.add_argument("--num-train-epochs", type=float, default=5.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--evaluation-strategy", default="epoch")
    parser.add_argument("--save-strategy", default="epoch")
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--metric-for-best-model", default="f1_micro")
    parser.add_argument("--greater-is-better", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--do-train", action="store_true")
    parser.add_argument("--do-eval", action="store_true")
    parser.add_argument("--do-predict", action="store_true")
    return parser.parse_args()


def get_datasets(train_file: str, validation_file: str, test_file: str) -> DatasetDict:
    data_files = {
        "train": train_file,
        "validation": validation_file,
        "test": test_file,
    }
    return load_dataset("json", data_files=data_files)


def preprocess_datasets(
    datasets: DatasetDict,
    tokenizer: AutoTokenizer,
    max_source_length: int,
    max_target_length: int,
) -> DatasetDict:
    padding = "max_length"

    def preprocess_function(examples: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        inputs = examples["text"]
        targets = examples["target_text"]
        model_inputs = tokenizer(
            inputs,
            max_length=max_source_length,
            padding=padding,
            truncation=True,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=max_target_length,
                padding=padding,
                truncation=True,
            )
        labels_ids = labels["input_ids"]
        labels_ids = [
            [token_id if token_id != tokenizer.pad_token_id else -100 for token_id in label]
            for label in labels_ids
        ]
        model_inputs["labels"] = labels_ids
        return model_inputs

    return datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=datasets["train"].column_names,
    )


def normalise_text(text: str, tokenizer: AutoTokenizer) -> str:
    if text is None:
        return ""
    clean = text
    for tok in filter(None, [tokenizer.pad_token, tokenizer.eos_token, tokenizer.bos_token, "</s>", "<s>"]):
        clean = clean.replace(tok, " ")
    return " ".join(clean.split())


def extract_triplets(text: str) -> List[Tuple[str, str, str]]:
    tokens = text.split()
    triplets: List[Tuple[str, str, str]] = []
    idx = 0
    while idx < len(tokens):
        if tokens[idx] != TRIPLET_TOKEN:
            idx += 1
            continue
        idx += 1
        subj_tokens: List[str] = []
        while idx < len(tokens) and tokens[idx] != SUBJ_TOKEN:
            subj_tokens.append(tokens[idx])
            idx += 1
        if idx >= len(tokens) or tokens[idx] != SUBJ_TOKEN:
            break
        idx += 1
        obj_tokens: List[str] = []
        while idx < len(tokens) and tokens[idx] != OBJ_TOKEN:
            obj_tokens.append(tokens[idx])
            idx += 1
        if idx >= len(tokens) or tokens[idx] != OBJ_TOKEN:
            break
        idx += 1
        rel_tokens: List[str] = []
        while idx < len(tokens) and tokens[idx] != TRIPLET_TOKEN:
            rel_tokens.append(tokens[idx])
            idx += 1
        subject = " ".join(subj_tokens).strip()
        obj = " ".join(obj_tokens).strip()
        relation = " ".join(rel_tokens).strip()
        if subject and obj and relation:
            triplets.append((subject, obj, relation))
    return triplets


def compute_per_class_metrics(
    preds: List[List[Tuple[str, str, str]]],
    labels: List[List[Tuple[str, str, str]]],
) -> Dict[str, float]:
    metrics: Dict[str, Dict[str, int]] = {}

    def to_norm(triple: Tuple[str, str, str]) -> Tuple[str, str, str]:
        return tuple(part.lower() for part in triple)

    def ensure_class(rel: str) -> Dict[str, int]:
        if rel not in metrics:
            metrics[rel] = {"tp": 0, "fp": 0, "fn": 0}
        return metrics[rel]

    for predicted, gold in zip(preds, labels):
        pred_map = {to_norm(triple): triple[2] for triple in predicted}
        gold_map = {to_norm(triple): triple[2] for triple in gold}
        predicted_set = set(pred_map.keys())
        gold_set = set(gold_map.keys())

        for triple in predicted_set:
            rel = pred_map[triple]
            counts = ensure_class(rel)
            if triple in gold_set:
                counts["tp"] += 1
            else:
                counts["fp"] += 1
        for triple in gold_set:
            rel = gold_map[triple]
            counts = ensure_class(rel)
            if triple not in predicted_set:
                counts["fn"] += 1

    summary: Dict[str, float] = {}
    total_tp = total_fp = total_fn = 0
    for relation, counts in metrics.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        total_tp += tp
        total_fp += fp
        total_fn += fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        key = relation.replace(" ", "_")
        summary[f"precision_{key}"] = precision
        summary[f"recall_{key}"] = recall
        summary[f"f1_{key}"] = f1

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )
    summary["precision_micro"] = micro_precision
    summary["recall_micro"] = micro_recall
    summary["f1_micro"] = micro_f1
    return summary


def build_compute_metrics(tokenizer: AutoTokenizer):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=False)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False)

        cleaned_preds = [normalise_text(text, tokenizer) for text in decoded_preds]
        cleaned_labels = [normalise_text(text, tokenizer) for text in decoded_labels]

        pred_triplets = [extract_triplets(text) for text in cleaned_preds]
        gold_triplets = [extract_triplets(text) for text in cleaned_labels]

        return compute_per_class_metrics(pred_triplets, gold_triplets)

    return compute_metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        eval_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better or None,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        seed=args.seed,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        load_best_model_at_end=True,
        do_train=args.do_train,
        do_eval=args.do_eval,
        do_predict=args.do_predict,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)

    raw_datasets = get_datasets(args.train_file, args.validation_file, args.test_file)
    tokenized_datasets = preprocess_datasets(
        raw_datasets,
        tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets.get("train"),
        eval_dataset=tokenized_datasets.get("validation"),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
    )

    metrics = {}

    if training_args.do_train:
        LOGGER.info("Starting training")
        train_result = trainer.train()
        metrics.update(train_result.metrics)
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        LOGGER.info("Running validation evaluation")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    if training_args.do_predict:
        LOGGER.info("Running test evaluation")
        test_metrics = trainer.predict(tokenized_datasets.get("test"), metric_key_prefix="test")
        trainer.log_metrics("test", test_metrics.metrics)
        trainer.save_metrics("test", test_metrics.metrics)


if __name__ == "__main__":
    main()
