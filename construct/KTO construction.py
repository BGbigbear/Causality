import json
import random

from tqdm import tqdm
from construct.judge_same_event import is_similar_event

open_filename = '../data/Alpaca/merge.json'
instruction_filename = '../data/fewshot/causality_train1_analysis_rougeSFT_full_0.json'  # 包含 instruction 的数据文件
write_filename = '../data/Alpaca/KTOdatas.json'


def classify_list(bad_data):
    classified_data = []
    visited = set()  # 用于记录已处理的元素索引，避免重复分组

    for i, item in enumerate(bad_data):
        if i in visited:
            continue  # 跳过已分组的元素

        # 创建一个新的子列表用于存储相似的元素
        similar_group = [item]
        visited.add(i)
        for j in range(i + 1, len(bad_data)):
            if j not in visited and is_similar_event(str(item), str(bad_data[j])):
                similar_group.append(bad_data[j])
                visited.add(j)

        # 将相似组加入分类列表
        classified_data.append(similar_group)

    return classified_data


def create_limited_causality_list(bad_data, max_length):
    normalized_bad_data = [
        sublist if isinstance(sublist, list) else [sublist]
        for sublist in bad_data
    ]
    selected_pairs = [random.choice(sublist) for sublist in normalized_bad_data if sublist]
    causality_list = random.sample(selected_pairs, min(max_length, len(selected_pairs)))
    return causality_list

# 读取 instruction 文件并构建以 document_id 为键的字典
with open(instruction_filename, "r", encoding="utf-8") as f:
    instructions_data = json.load(f)

# 从 instructions_data 提取 document_id 和 user 的 content
instructions_dict = {}
for item in instructions_data:
    document_id = item.get("document_id")
    if document_id and "analysis" in item:
        # 提取 analysis 中 role 为 "user" 的 content
        user_content = next(
            (entry["content"] for entry in item["analysis"] if entry["role"] == "user"), ""
        )
        instructions_dict[document_id] = user_content
print(len(instructions_dict))
# 读取主数据文件
with open(open_filename, "r", encoding="utf-8") as f:
    data = json.load(f)

KTOdatas = []
for document in tqdm(data, desc="Processing:"):

    causality_list = create_limited_causality_list(classify_list(document["BAD"]), 3)
    if not causality_list: continue
    document_id = document["document_id"]
    # 从 instructions_dict 中获取 instruction
    instruction = instructions_dict.get(document_id, "")

    KTOdata = {
        "instruction": instruction,
        "input": "",
        "output": "```json\n{\"causality_list\":" + json.dumps(causality_list,ensure_ascii=False) + "}\n```",
        "kto_tag": "false"
    }
    KTOdatas.append(KTOdata)

# 将 KTOdatas 写入 JSON 文件
with open(write_filename, "w", encoding="utf-8") as outfile:
    json.dump(KTOdatas, outfile, ensure_ascii=False, indent=4)

print("KTOdatas has been written to KTOdatas.json")
