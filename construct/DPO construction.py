import json
import random
import re
from deepdiff import DeepDiff

# 文件路径
open_filename = '../data/Alpaca/merge.json'
instruction_filename = '../data/fewshot/causality_train1_analysis_rougeSFT_full_0.json'
standard_filename= '../data/reconstruction/merge1_1.json'
write_filename = '../data/Alpaca/DPOdatas.json'
# 使用示例
def generate_unique_random_result(merged, previous_results, max_attempts=10):
    attempt = 0
    while attempt < max_attempts:
        # 每次尝试生成一个新 result
        result = [{"Valid": json.loads(valid), "Pred": random.choice(preds)} for valid, preds in merged.items()]
        if result not in previous_results:
            previous_results.append(result)
            return result
        attempt += 1
    return None
# 读取目标 instruction 文件
with open(instruction_filename, "r", encoding="utf-8") as f:
    instructions_data = json.load(f)

# 读取主数据文件
with open(open_filename, "r", encoding="utf-8") as f:
    data = json.load(f)
with open(standard_filename, "r", encoding="utf-8") as f:
    standard = json.load(f)
user_contents_dict = {
    item["document_id"]: next(
        (entry["content"] for entry in item.get("analysis", []) if entry.get("role") == "user"), ""
    )
    for item in instructions_data
    if isinstance(item, dict) and "document_id" in item
}
standard_dict = {
    item["document_id"]: item.get("causality_list", [])
    for item in standard
    if isinstance(item, dict) and "document_id" in item
}
output_dict = {
    item["document_id"]: item.get("MID", [])
    for item in data
    if isinstance(item, dict) and "document_id" in item
}
combined_list = []
# 遍历用户内容字典
for document_id, user_content in user_contents_dict.items():
    # 在标准字典中查找
    causality_list = standard_dict.get(document_id, [])

    # 在输出字典中查找
    mid = output_dict.get(document_id, [])

    # 合并信息到新的字典
    combined_entry = {
        "document_id": document_id,
        "user_content": user_content,
        "causality_list": causality_list,
        "MID": mid
    }

    # 添加到新字典列表中
    combined_list.append(combined_entry)
DPODatas=[]
for document in combined_list:
    if not document["MID"]:continue
    merged = {}
    # 合并相同的 Valid，并收集对应的 Pred
    for item in document["MID"]:
        valid = json.dumps(item["Valid"], sort_keys=True)  # 将 Valid 转为字符串以便作为字典的唯一标识
        pred = item["Pred"]

        if valid not in merged:
            merged[valid] = [pred]
        else:
            merged[valid].append(pred)

    # 随机选择一个 Pred 保留
    previous_results=[]
    for _ in range(3):
        new_result = generate_unique_random_result(merged, previous_results)
        if new_result is None:
            break
    print(len(previous_results))
    for i in range(3):
        if i>=len(previous_results):
            break
        rejected_list=[]
        for causality in document["causality_list"]:
            causality_valid = causality  # 将 Valid 转为字符串
            found = False

            # 检查 result 中的 Valid
            for res in previous_results[i]:
                res_valid = res["Valid"]
                if not DeepDiff(causality_valid, res_valid):
                    # 找到对应的 Valid，添加 Pred 到 rejected_list
                    rejected_list.append(res["Pred"])
                    found = True
                    break

            if not found:
               # 如果没有找到，将整个元素添加到 rejected_list
               rejected_list.append(causality)

        dpodata={
            "text":document["user_content"],
            "input":"",
            "chosen":"```json\n{\"causality_list\":" + json.dumps(document["causality_list"],ensure_ascii=False) + "}\n```",
            "rejected":"```json\n{\"causality_list\":" + json.dumps(rejected_list,ensure_ascii=False) + "}\n```"
        }
        DPODatas.append(dpodata)




print(len(DPODatas))
with open(write_filename, "w", encoding="utf-8") as outfile:
    json.dump(DPODatas, outfile, ensure_ascii=False, indent=4)
