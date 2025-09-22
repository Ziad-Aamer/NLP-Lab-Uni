import json
import csv
import os

few_shots = (True, """
Determine which option can be inferred from the give sentence.

Example 1:
Sentence: A novel SCN5A mutation manifests as a malignant form of long QT syndrome with perinatal onset of tachycardia/bradycardia.

Options:
A. bradycardia and SCN5A are positively correlated.
B. bradycardia and SCN5A are negatively correlated.
C. bradycardia and SCN5A are associated.
D. bradycardia and SCN5A have no known relation.
Which option can be inferred from the given sentence?

Option: C

Example 2:
Sentence: OBJECTIVE: Congenital long QT syndrome (LQTS) with in utero onset of the rhythm disturbances is associated with a poor prognosis. In this study we investigated a newborn patient with fetal bradycardia, 2:1 atrioventricular block and ventricular tachycardia soon after birth. METHODS: Mutational analysis and DNA sequencing were conducted in a newborn. The 2:1 atrioventricular block improved to 1:1 conduction only after intravenous lidocaine infusion or a high dose of mexiletine, which also controlled the ventricular tachycardia. RESULTS: A novel, spontaneous LQTS-3 mutation was identified in the transmembrane segment 6 of domain IV of the Na(v)1.5 cardiac sodium channel, with a G-->A substitution at codon 1763, which changed a valine (GTG) to a methionine (ATG). The proband was heterozygous but the mutation was absent in the parents and the sister. Expression of this mutant channel in tsA201 mammalian cells by site-directed mutagenesis revealed a persistent tetrodotoxin-sensitive but lidocaine-resistant current that was associated with a positive shift of the steady-state inactivation curve, steeper activation curve and faster recovery from inactivation. We also found a similar electrophysiological profile for the neighboring V1764M mutant. But, the other neighboring I1762A mutant had no persistent current and was still associated with a positive shift of inactivation. CONCLUSIONS: These findings suggest that the Na(v)1.5/V1763M channel dysfunction and possible neighboring mutants contribute to a persistent inward current due to altered inactivation kinetics and clinically congenital LQTS with perinatal onset of arrhythmias that responded to lidocaine and mexiletine.
Options:
A. bradycardia and V1763M are positively correlated.
B. bradycardia and V1763M are negatively correlated.
C. bradycardia and V1763M are associated.
D. bradycardia and V1763M have no known relation.
Which option can be inferred from the given sentence?

Option: D


Example 3:
Sentence: AIMS: To elucidate how the nicotinic acetylcholine receptors expressed on bronchial and oral epithelial cells targeted by the tobacco nitrosamine (4-(methylnitrosamino)-1-(3-pyridyl)-1-butanone) (NNK) facilitate carcinogenic transformation. MAIN METHODS: Since NNK-dependent transformation can be abolished by the nicotinergic secreted mammalian Ly-6/urokinase plasminogen activator receptor related protein-1 (SLURP-1), we compared effects of NNK and recombinant (r)SLURP-1 on the expression of genes related to tumorigenesis in human immortalized bronchial and oral epithelial cell lines BEP2D and Het-1A, respectively. KEY FINDINGS: NNK stimulated expression of oncogenic genes, including MYB and PIK3CA in BEP2D, ETS1, NRAS and SRC in Het-1A, and AKT1, KIT and RB1 in both cell types, which could be abolished in the presence of rSLURP-1. Other cancer-related genes whose upregulation by NNK was abolishable by rSLURP-1 were the growth factors EGF in BEP2D cells and HGF in Het-1A cells, and the transcription factors CDKN2A and STAT3 (Het-1A only). NNK also upregulated the anti-apoptotic BCL2 (Het-1A) and downregulated the pro-apoptotic TNF (Het-1A), BAX and CASP8 (BEP2D), all of which could be abolished, in part, by rSLURP-1. NNK decreased expression of the CTNNB1 gene encoding the intercellular adhesion molecule beta-catenin (BEP2D), as well as tumor suppressors CDKN3 and FOXD3 in BEP2D cells and SERPINB5 in Het-1A cells. These pro-oncogenic effects of NNK were abolished by rSLURP-1 that also upregulated RUNX3. SIGNIFICANCE: The obtained results identified target genes for both NNK and SLURP-1 and shed light on the molecular mechanism of their reciprocal effects on tumorigenic transformation of bronchial and oral epithelial cells.

Options:
A. NNK and PIK3CA are positively correlated.
B. NNK and PIK3CA are negatively correlated.
C. NNK and PIK3CA are associated.
D. NNK binds to PIK3CA.
E. NNK and PIK3CA have no known relation.
Which option can be inferred from the given sentence?

Option: E

Now, solve this case:
""")
# "MINI": "mini-biored.JSON"
DATASETS = {
    "Dev": "../BioRED/dataset/Dev.BioC.JSON",
    "Test": "../BioRED/dataset/Test.BioC.JSON",
    "Train": "../BioRED/dataset/Train.BioC.JSON",
    "MINI": "./mini-biored.JSON"
    
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
                        
                    if few_shots[0]:
                        qa4re_prompt = (
                            few_shots[1] +
                            f"Sentence: {passage_text}\n"
                            f"\nOptions:\n" +
                            "\n".join(qa_options) +
                            "\nWhich option can be inferred from the given sentence?\nOption:"
                        )
                    else:
                        qa4re_prompt = (
                            f"Determine which option can be inferred from the give sentence.\n"
                            f"Sentence: {passage_text}\n"
                            f"\nOptions:\n" +
                            "\n".join(qa_options) +
                            "\nWhich option can be inferred from the given sentence?\nOption:"
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
        
        if few_shots[0]:
            qa4re_csv = f"{prompts_dir}/qa4re_prompts_with_gold_{name.upper()}_FS.csv"
        else:
            qa4re_csv = f"{prompts_dir}/qa4re_prompts_with_gold_{name.upper()}.csv"
        os.makedirs(prompts_dir, exist_ok=True)
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
