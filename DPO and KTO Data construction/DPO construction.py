import json

open_filename = '../data/Alpaca/re.json'
instruction_filename = '../result/causality_test1_analysis_rouge_full.json'  # 包含 instruction 的数据文件
write_filename = '../data/Alpaca/DPOdatas.json'

# 读取目标 instruction 文件
with open(instruction_filename, "r", encoding="utf-8") as f:
    instructions_data = json.load(f)

# 读取主数据文件
with open(open_filename, "r", encoding="utf-8") as f:
    data = json.load(f)

DPOdatas = []
for document in data:
    document_id = document["document_id"]
    rejected_list = [item["Pred"] for item in document["MID"]]
    chosen_list = [item["Valid"] for item in document["MID"]]
    if not rejected_list:
        if not chosen_list:
            continue
    # 查找与当前 document_id 匹配的 instruction
    instruction = ""
    for item in instructions_data:
        if item.get("document_id") == document_id:
            instruction = next(
                (entry["content"] for entry in item.get("analysis", []) if entry["role"] == "user"), ""
            )
            break  # 找到匹配项后立即停止查找

    dpodata = {
        "instruction": instruction,
        "input": document["text"],
        "chosen": f"```json\n{json.dumps(chosen_list, ensure_ascii=False, indent=4)}\n```",
        "rejected": f"```json\n{json.dumps(rejected_list, ensure_ascii=False, indent=4)}\n```"
    }
    DPOdatas.append(dpodata)

# 将 DPOdatas 写入 JSON 文件
with open(write_filename, "w", encoding="utf-8") as outfile:
    json.dump(DPOdatas, outfile, ensure_ascii=False, indent=4)

print("DPOdatas has been written to DPOdatas.json")
