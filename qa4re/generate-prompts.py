import json

input_path = "../BioRED/dataset/Dev.BioC.JSON"
vanilla_output_path = "vanilla_prompts.txt"
qa4re_output_path = "qa4re_prompts.txt"


# Canonical entity classes for mapping (from BioRED)
ENTITY_TYPE_MAP = {
    'GeneOrGeneProduct': 'Gene',
    'DiseaseOrPhenotypicFeature': 'Disease',
    'ChemicalEntity': 'Chemical',
    'SequenceVariant': 'Variant',
    'OrganismTaxon': 'Species',
    'CellLine': 'CellLine'
}

# 8 BioRED allowed entity pairings (order-insensitive, as per paper)
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

# TODO: Double-check these relation types against BioRED  (e.g. Chemical-Chemical looks to have more types)
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

relation_type_templates = {
    "Association": "{} and {} are associated.",
    "Positive_Correlation": "{} and {} are positively correlated.",
    "Negative_Correlation": "{} and {} are negatively correlated.",
    # You can add more types as needed.
    "No_Relation": "{} and {} have no known relation."
}
relation_type_list = [
    "Association", "Positive_Correlation", "Negative_Correlation", "No_Relation"
]

def get_entity_text_by_id(annotations, identifier):
    # Find the first annotation whose 'infons.identifier' matches
    for ann in annotations:
        if ann['infons']['identifier'] == identifier:
            return ann['text']
    return identifier # fallback: use id string


# Helper: get canonical entity class from annotation dict
def get_entity_class(anno):
    return ENTITY_TYPE_MAP.get(anno['infons']['type'], anno['infons']['type'])

# Helper: find annotation by id in annotation list
def get_annotation_by_id(annotations, identifier):
    for ann in annotations:
        if ann['infons']['identifier'] == identifier:
            return ann
    return None

def main():
    with open(input_path, encoding='utf-8') as f:
        data = json.load(f)

    vanilla_prompts = []
    qa4re_prompts = []

    for doc in data['documents']:
        # Pre-index all annotations by id across all passages for this doc
        all_annotations = []
        for p in doc['passages']:
            all_annotations.extend(p.get('annotations', []))
        # For each passage, check which relations have both entities in that passage
        for passage in doc['passages']:
            passage_text = passage['text']
            passage_annos = passage.get('annotations', [])

            # Collect the set of identifiers present in this passage
            passage_entity_ids = {a['infons']['identifier'] for a in passage_annos}

            # For each relation, if both entities present in this passage, generate prompts
            for rel in doc.get('relations', []):
                info = rel['infons']
                ent1_id, ent2_id = info['entity1'], info['entity2']

                if ent1_id in passage_entity_ids and ent2_id in passage_entity_ids:
                    ann1 = get_annotation_by_id(all_annotations, ent1_id)
                    ann2 = get_annotation_by_id(all_annotations, ent2_id)
                    if not (ann1 and ann2):
                        continue  # Skip if we can't resolve both entities

                    ent1_txt = ann1['text']
                    ent2_txt = ann2['text']
                    ent1_type = get_entity_class(ann1)
                    ent2_type = get_entity_class(ann2)

                    entity_pair = frozenset([ent1_type, ent2_type])

                    allowed_relation_types = ALLOWED_RELATION_TYPES.get(entity_pair, [])
                    if not allowed_relation_types:
                        print(f"Skipping entity pair with no allowed relations: {ent1_type}, {ent2_type}")
                        # continue  # skip if not valid entity pair

                    # For vanilla prompt: use canonical label
                    vanilla_options = allowed_relation_types + ["No_Relation"]
                    
                    vanilla = (
                        f"Given the following passage and two entities, classify their relationship.\n"
                        f"Passage: {passage_text}\n"
                        f"Entity 1: {ent1_txt}\n"
                        f"Entity 2: {ent2_txt}\n"
                        f"Possible relations: {', '.join(vanilla_options)}\n"
                        f"Relationship:\n"
                    )
                    vanilla_prompts.append(vanilla)

                    # For QA4RE: Use templates for only the allowed types (plus No_Relation)
                    qa_options = []
                    letter = ord('A')
                    for rtype in allowed_relation_types:
                        template = RELATION_TYPE_TEMPLATES[rtype]
                        qa_options.append(f"{chr(letter)}. " + template.format(ent1_txt, ent2_txt))
                        letter += 1
                    qa_options.append(f"{chr(letter)}. {ent1_txt} and {ent2_txt} have no known relation.")
                    qa4re = (
                        f"Determine which option can be inferred from the following passage.\n"
                        f"Passage: {passage_text}\n"
                        f"Options:\n" +
                        "\n".join(qa_options) +
                        "\nWhich option can be inferred?\nOption:\n"
                    )
                    qa4re_prompts.append(qa4re)

    # --- Write to files ---
    with open(vanilla_output_path, "w", encoding="utf-8") as f:
        for prompt in vanilla_prompts:
            f.write(prompt.strip() + "\n\n")

    with open(qa4re_output_path, "w", encoding="utf-8") as f:
        for prompt in qa4re_prompts:
            f.write(prompt.strip() + "\n\n")

    print(f"Wrote {len(vanilla_prompts)} vanilla prompts to {vanilla_output_path}")
    print(f"Wrote {len(qa4re_prompts)} QA4RE prompts to {qa4re_output_path}")

if __name__ == "__main__":
    main()
