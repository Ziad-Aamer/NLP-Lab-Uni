import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import csv

import json
import os
import csv
from itertools import combinations


def convert_bioc_to_tsv(input_dir, output_dir, splits=["Train", "Dev", "Test"]):
    os.makedirs(output_dir, exist_ok=True)

    for split in splits:
        input_file = os.path.join(input_dir, f"{split}.BioC.JSON")
        output_file = os.path.join(output_dir, f"{split.lower()}.tsv")
        print(f"Converting {input_file} → {output_file}")

        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        samples = []

        for doc in data["documents"]:
            # Step 1: Extract relations as (entity1_id, entity2_id) → relation_type
            relation_lookup = {}
            for rel in doc.get("relations", []):
                ent1_id = rel["infons"].get("entity1")
                ent2_id = rel["infons"].get("entity2")
                rel_type = rel["infons"].get("type", "No-Relation")

                # Bi-directional match
                relation_lookup[(ent1_id, ent2_id)] = rel_type
                relation_lookup[(ent2_id, ent1_id)] = rel_type

            # Step 2: Process passages
            for passage in doc.get("passages", []):
                passage_text = passage.get("text", "").strip()
                annotations = passage.get("annotations", [])

                # Map entity ID → name
                entity_map = {}
                for ann in annotations:
                    identifier = ann["infons"].get("identifier")
                    name = ann["text"]
                    if identifier and name:
                        entity_map[identifier] = name.strip()

                entity_ids = list(entity_map.keys())

                # Step 3: All unique entity pairs in this passage
                for id1, id2 in combinations(entity_ids, 2):
                    if (id1, id2) in relation_lookup:
                        rel_type = relation_lookup[(id1, id2)]
                        samples.append([
                            passage_text,
                            entity_map[id1],
                            entity_map[id2],
                            rel_type
                        ])

        # Write TSV
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

