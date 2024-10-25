import json

from tqdm import tqdm

from config.configuration import causality_file, test_file, global_prompts
from util.rouge import top_similar_text

with (
    open(causality_file, 'r', encoding='utf-8') as f_causality,
    open(test_file, 'r', encoding='utf-8') as f_test
):
    causality_data = json.load(f_causality)
    test_data = json.load(f_test)

data = []
for i, doc in tqdm(enumerate(test_data), desc="Constructing", total=len(test_data)):
    few_shot = ""
    shots = {"Event_extraction_examples": top_similar_text(doc['text'], causality_data, top_k=3)}
    few_shot += json.dumps(shots, ensure_ascii=False, indent=2)
    data.append({
        "instruction": f"{few_shot}\n\n{global_prompts[0]}",
        "input": f"text: {doc['text']}",
        "output": ""
    })

with open('../data/fewshot/train1_plus2.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

