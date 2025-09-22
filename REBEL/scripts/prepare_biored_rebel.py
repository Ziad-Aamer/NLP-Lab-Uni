#!/usr/bin/env python
"""Preprocess BioRED BioC JSON into REBEL-friendly seq2seq data."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

LOGGER = logging.getLogger(__name__)

TRIPLET_TOKEN = "<triplet>"
SUBJ_TOKEN = "<subj>"
OBJ_TOKEN = "<obj>"


def normalise_whitespace(text: str) -> str:
    return " ".join(text.split())


def build_entity_lexicon(passages: Iterable[Dict]) -> Dict[str, str]:
    """Map entity identifiers and annotation ids to display texts."""
    lexicon: Dict[str, str] = {}
    for passage in passages:
        for ann in passage.get("annotations", []):
            text = normalise_whitespace(ann.get("text", "").strip())
            if not text:
                continue
            ann_id = ann.get("id")
            if ann_id and ann_id not in lexicon:
                lexicon[ann_id] = text
            identifier = ann.get("infons", {}).get("identifier")
            if identifier:
                for key in identifier.split(","):
                    key = key.strip()
                    if key and key not in lexicon:
                        lexicon[key] = text
    return lexicon


def build_triplets(doc: Dict) -> List[Tuple[str, str, str]]:
    lexicon = build_entity_lexicon(doc.get("passages", []))
    triplets: List[Tuple[str, str, str]] = []
    for rel in doc.get("relations", []):
        infons = rel.get("infons", {})
        subj_id = infons.get("entity1")
        obj_id = infons.get("entity2")
        rel_type = infons.get("type")
        subj_text = normalise_whitespace(lexicon.get(subj_id, ""))
        obj_text = normalise_whitespace(lexicon.get(obj_id, ""))
        if not subj_text or not obj_text or not rel_type:
            LOGGER.debug("Skipping relation %s due to missing fields", rel.get("id"))
            continue
        triplets.append((subj_text, obj_text, rel_type))
    return triplets


def doc_to_example(doc: Dict) -> Dict:
    triplets = build_triplets(doc)
    if not triplets:
        return {}
    text_parts = []
    for passage in doc.get("passages", []):
        passage_text = passage.get("text", "")
        if passage_text:
            text_parts.append(normalise_whitespace(passage_text))
    source_text = " \n ".join(text_parts)
    target_chunks = [
        f"{TRIPLET_TOKEN} {subj} {SUBJ_TOKEN} {obj} {OBJ_TOKEN} {rel}"
        for subj, obj, rel in triplets
    ]
    target_text = " ".join(target_chunks)
    return {
        "doc_id": doc.get("id"),
        "text": source_text,
        "target_text": target_text,
        "relations": [
            {"subject": subj, "object": obj, "relation": rel}
            for subj, obj, rel in triplets
        ],
    }


def process_split(input_path: Path, output_path: Path) -> int:
    with input_path.open() as f:
        data = json.load(f)
    examples = []
    for doc in data.get("documents", []):
        example = doc_to_example(doc)
        if example:
            examples.append(example)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as out_f:
        for example in examples:
            out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
    return len(examples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("BioRED/dataset"),
        help="Directory containing BioC JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("REBEL/data"),
        help="Directory to write JSONL datasets",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["Train", "Dev", "Test"],
        help="Dataset splits to process",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    for split in args.splits:
        input_path = args.input_dir / f"{split}.BioC.JSON"
        output_path = args.output_dir / f"biored_rebel_{split.lower()}.jsonl"
        LOGGER.info("Processing %s -> %s", input_path, output_path)
        count = process_split(input_path, output_path)
        LOGGER.info("Wrote %d examples", count)


if __name__ == "__main__":
    main()
