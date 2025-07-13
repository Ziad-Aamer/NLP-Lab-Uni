import json
import csv
import os

# "MINI": "mini-biored.JSON"
DATASETS = {
    "Dev": "../BioRED/dataset/Dev.BioC.JSON",
    "Test": "../BioRED/dataset/Test.BioC.JSON",
    "Train": "../BioRED/dataset/Train.BioC.JSON"
}

ENTITY_TYPE_MAP = {
    'GeneOrGeneProduct': 'Gene',
    'DiseaseOrPhenotypicFeature': 'Disease',
    'ChemicalEntity': 'Chemical',
    'SequenceVariant': 'Variant',
    'OrganismTaxon': 'Species',
    'CellLine': 'CellLine'
}

ALLOWED_PAIRINGS = {
    frozenset(['Disease', 'Gene']),
    frozenset(['Disease', 'Chemical']),
    frozenset(['Gene', 'Chemical']),
    frozenset(['Gene', 'Gene']),
    frozenset(['Disease', 'Variant']),
    frozenset(['Chemical', 'Variant']),
    frozenset(['Chemical', 'Chemical']),
    frozenset(['Variant', 'Variant']),
}

ALLOWED_RELATION_TYPES = {
    frozenset(['Disease', 'Variant']): [
        "Positive Correlation", "Negative Correlation", "Association"
    ],
    frozenset(['Disease', 'Gene']): [
        "Positive Correlation", "Negative Correlation", "Association"
    ],
    frozenset(['Disease', 'Chemical']): [
        "Positive Correlation", "Negative Correlation", "Association", "Drug Interaction", "Cotreatment"
    ],
    frozenset(['Chemical', 'Chemical']): [
        "Association", "Drug Interaction", "Cotreatment", "Comparison"
    ],
    frozenset(['Chemical', 'Variant']): [
        "Association", "Conversion"
    ],
    frozenset(['Variant', 'Variant']): [
        "Association", "Comparison", "Conversion"
    ],
    frozenset(['Gene', 'Chemical']): [
        "Positive Correlation", "Negative Correlation", "Association", "Bind"
    ],
    frozenset(['Gene', 'Gene']): [
        "Positive Correlation", "Negative Correlation", "Association", "Bind", "Comparison"
    ]
}

RELATION_TYPE_TEMPLATES = {
    "Positive Correlation": "{} and {} are positively correlated.",
    "Negative Correlation": "{} and {} are negatively correlated.",
    "Association": "{} and {} are associated.",
    "Bind": "{} binds to {}.",
    "Drug Interaction": "{} and {} have a drug interaction.",
    "Cotreatment": "{} and {} are used in cotreatment.",
    "Comparison": "{} and {} are compared.",
    "Conversion": "{} is converted to {}."
}

def get_entity_class(anno):
    return ENTITY_TYPE_MAP.get(anno['infons']['type'], anno['infons']['type'])

def get_annotation_by_id(annotations, identifier):
    for ann in annotations:
        if ann['infons']['identifier'] == identifier:
            return ann
    return None

def process_biored_json(input_path):
    with open(input_path, encoding='utf-8') as f:
        data = json.load(f)

    vanilla_rows = []
    qa4re_rows = []

    for doc in data['documents']:
        all_annotations = []
        for p in doc['passages']:
            all_annotations.extend(p.get('annotations', []))
        for passage in doc['passages']:
            passage_text = passage['text']
            passage_annos = passage.get('annotations', [])
            passage_entity_ids = {a['infons']['identifier'] for a in passage_annos}
            for rel in doc.get('relations', []):
                info = rel['infons']
                ent1_id, ent2_id = info['entity1'], info['entity2']
                gold_relation = info['type']
                if ent1_id in passage_entity_ids and ent2_id in passage_entity_ids:
                    ann1 = get_annotation_by_id(all_annotations, ent1_id)
                    ann2 = get_annotation_by_id(all_annotations, ent2_id)
                    if not (ann1 and ann2):
                        continue
                    ent1_txt = ann1['text']
                    ent2_txt = ann2['text']
                    ent1_type = get_entity_class(ann1)
                    ent2_type = get_entity_class(ann2)
                    entity_pair = frozenset([ent1_type, ent2_type])
                    allowed_relation_types = ALLOWED_RELATION_TYPES.get(entity_pair, [])
                    if not allowed_relation_types:
                        continue

                    # --- VANILLA PROMPT ---
                    vanilla_options = allowed_relation_types + ["No Relation"]
                    vanilla_prompt = (
                        f"Given the following passage and two entities, classify their relationship.\n"
                        f"Passage: {passage_text}\n"
                        f"Entity 1: {ent1_txt}\n"
                        f"Entity 2: {ent2_txt}\n"
                        f"Possible relations: {', '.join(vanilla_options)}\n"
                        f"Relationship:"
                    )
                    vanilla_gold = gold_relation if gold_relation in allowed_relation_types else "No Relation"
                    vanilla_rows.append({
                        "vanilla_prompt": vanilla_prompt,
                        "gold_label": vanilla_gold
                    })

                    # --- QA4RE PROMPT ---
                    qa_options = []
                    gold_option_letter = None
                    letter = ord('A')
                    for rtype in allowed_relation_types:
                        template = RELATION_TYPE_TEMPLATES[rtype]
                        option_str = template.format(ent1_txt, ent2_txt)
                        qa_options.append(f"{chr(letter)}. {option_str}")
                        if rtype == gold_relation:
                            gold_option_letter = chr(letter)
                        letter += 1
                    qa_options.append(f"{chr(letter)}. {ent1_txt} and {ent2_txt} have no known relation.")
                    if gold_option_letter is None:
                        gold_option_letter = chr(letter)
                    qa4re_prompt = (
                        f"Determine which option can be inferred from the following passage.\n"
                        f"Passage: {passage_text}\n"
                        f"Options:\n" +
                        "\n".join(qa_options) +
                        "\nWhich option can be inferred?\nOption:"
                    )
                    qa4re_rows.append({
                        "qa4re_prompt": qa4re_prompt,
                        "gold_option_letter": gold_option_letter
                    })
    return vanilla_rows, qa4re_rows

def main():
    for name, path in DATASETS.items():
        vanilla_rows, qa4re_rows = process_biored_json(path)
        prompts_dir = "prepared_prompt_files"
        vanilla_csv = f"{prompts_dir}/vanilla_prompts_with_gold_{name.upper()}.csv"
        qa4re_csv = f"{prompts_dir}/qa4re_prompts_with_gold_{name.upper()}.csv"
        with open(vanilla_csv, "w", encoding="utf-8", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["vanilla_prompt", "gold_label"])
            writer.writeheader()
            for row in vanilla_rows:
                writer.writerow(row)
        with open(qa4re_csv, "w", encoding="utf-8", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["qa4re_prompt", "gold_option_letter"])
            writer.writeheader()
            for row in qa4re_rows:
                writer.writerow(row)
        print(f"Wrote {len(vanilla_rows)} vanilla prompts with gold labels to {vanilla_csv}")
        print(f"Wrote {len(qa4re_rows)} QA4RE prompts with gold letters to {qa4re_csv}")

if __name__ == "__main__":
    main()
