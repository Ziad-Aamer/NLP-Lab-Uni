import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import csv

def convert_bioc_to_tsv(input_dir, output_dir, splits=["Train", "Dev", "Test"]):
    for split in splits:
        input_file = os.path.join(input_dir, f"{split}.BioC.JSON")
        output_file = os.path.join(output_dir, f"{split.lower()}.tsv")
        print(f"Converting {input_file} → {output_file}")

        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        samples = []

        for doc in data["documents"]:
            passage_lookup = {}  # passage_id → text
            annotation_map = {}  # entity_id → (mention_text, offset, length, type)

            # Build passage and annotation map
            for passage in doc.get("passages", []):
                passage_text = passage["text"]
                offset = passage["offset"]

                for ann in passage.get("annotations", []):
                    ent_id = ann["infons"].get("identifier")
                    ent_type = ann["infons"].get("type")
                    text = ann["text"]
                    loc = ann["locations"][0]
                    abs_offset = offset + loc["offset"]  # position in whole document

                    annotation_map[ent_id] = {
                        "text": text,
                        "offset": abs_offset,
                        "length": loc["length"],
                        "type": ent_type,
                        "passage_text": passage_text,
                        "passage_offset": offset
                    }

            # Process relations
            for rel in doc.get("relations", []):
                rel_type = rel["infons"].get("type", "No-Relation")

                ent1_id = rel["infons"].get("entity1")
                ent2_id = rel["infons"].get("entity2")

                if ent1_id not in annotation_map or ent2_id not in annotation_map:
                    continue  # skip if missing

                ent1 = annotation_map[ent1_id]
                ent2 = annotation_map[ent2_id]

                # Use the passage where both entities appear (or fall back to ent1)
                passage_text = ent1["passage_text"]

                samples.append([
                    passage_text.strip(),
                    ent1["text"].strip(),
                    ent2["text"].strip(),
                    rel_type.strip()
                ])

        # Write to TSV
        with open(output_file, "w", encoding="utf-8", newline='') as out_f:
            writer = csv.writer(out_f, delimiter="\t")
            for row in samples:
                writer.writerow(row)

        print(f"Wrote {len(samples)} samples to {output_file}")

def convert_single_bioc_to_tsv(input_file, output_file):

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []

    for doc in data["documents"]:
        passage_lookup = {}
        annotation_map = {}

        for passage in doc.get("passages", []):
            passage_text = passage["text"]
            offset = passage["offset"]

            for ann in passage.get("annotations", []):
                ent_id = ann["infons"].get("identifier")
                ent_type = ann["infons"].get("type")
                text = ann["text"]
                loc = ann["locations"][0]
                abs_offset = offset + loc["offset"]

                annotation_map[ent_id] = {
                    "text": text,
                    "offset": abs_offset,
                    "length": loc["length"],
                    "type": ent_type,
                    "passage_text": passage_text,
                    "passage_offset": offset
                }

        for rel in doc.get("relations", []):
            rel_type = rel["infons"].get("type", "No-Relation")
            ent1_id = rel["infons"].get("entity1")
            ent2_id = rel["infons"].get("entity2")

            if ent1_id not in annotation_map or ent2_id not in annotation_map:
                continue

            ent1 = annotation_map[ent1_id]
            ent2 = annotation_map[ent2_id]
            passage_text = ent1["passage_text"]

            samples.append([
                passage_text.strip(),
                ent1["text"].strip(),
                ent2["text"].strip(),
                rel_type.strip()
            ])

    with open(output_file, "w", encoding="utf-8", newline='') as out_f:
        writer = csv.writer(out_f, delimiter="\t")
        for row in samples:
            writer.writerow(row)
    
    print(f"Wrote {len(samples)} samples to {output_file}")

