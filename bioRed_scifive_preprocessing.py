import json
from pathlib import Path
from typing import List, Tuple, Dict
from datasets import Dataset, DatasetDict

BIOC_SPLITS = {
    "train": Path("BioRED/Train.BioC.JSON"),
    "dev": Path("BioRED/Dev.BioC.JSON"),
    "test": Path("BioRED/Test.BioC.JSON")
}

def extract_entities(doc: Dict) -> Dict[str, str]:
    entity_map = {}
    for passage in doc.get("passages", []):
        for ann in passage.get("annotations", []):
            entity_id = ann["infons"]["identifier"]
            entity_map[entity_id] = ann["text"]
    return entity_map

def extract_relations(doc: Dict, entity_map: Dict[str, str]) -> List[str]:
    triples = []
    for rel in doc.get("relations", []):
        e1 = rel["infons"].get("entity1")
        e2 = rel["infons"].get("entity2")
        rtype = rel["infons"].get("type")
        if e1 in entity_map and e2 in entity_map:
            head = entity_map[e1]
            tail = entity_map[e2]
            triples.append(f"{head} | {rtype} | {tail}")
    return triples

def process_bioc_file(path: Path) -> List[Dict[str, str]]:
    with open(path) as f:
        bioc = json.load(f)

    data = []
    for doc in bioc["documents"]:
        full_text = " ".join([p["text"] for p in doc["passages"]])
        entity_map = extract_entities(doc)
        triples = extract_relations(doc, entity_map)
        input_text = f"extract relations: {full_text}"
        output_text = "\n".join(triples) if triples else "None"
        data.append({"input": input_text, "output": output_text})
    return data

# Process all splits and convert to HuggingFace Dataset
hf_datasets = {}
for split, path in BIOC_SPLITS.items():
    records = process_bioc_file(path)
    hf_datasets[split] = Dataset.from_list(records)

biored_dataset = DatasetDict(hf_datasets)

# Save to disk (optional)
biored_dataset.save_to_disk("biored_scifive_text2text")
