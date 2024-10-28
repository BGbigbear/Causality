import json
from tqdm import tqdm  # 导入 tqdm

from rouge_recall import final_sel

open_filename='../data/reconstruction/merge1_1.json'
write_filename='../data/Alpaca/merge1_1_plus.json'
#获取rouge值相似并随机选择3次的序列
sequence = final_sel()
print(len(sequence))
with open(open_filename, "r", encoding="utf-8") as f:
    data = json.load(f)
# 创建一个字典以便快速查找 data 中的文档
data_dict = {doc['document_id']: doc for doc in data}
final_result = []

# 使用 tqdm 创建进度条
for doc_id, selections in tqdm(sequence.items(), desc="Processing documents"):
    if doc_id in data_dict:
        # 获取与 doc_id 匹配的文档
        cur_doc = data_dict[doc_id]
        input_prompt = "text:" + cur_doc["text"]
        output_prompt = "```json\n{\"causality_list\":" + json.dumps(cur_doc["causality_list"], ensure_ascii=False) + "}\n```"
        same_prompt = "\n注意:\n1.只需要抽取包含“直接”因果关系的事件。如果某一事件，它和其他的事件之间不构成直接因果关系，则不需要包括在结果中。每一组因果关系，必须包含一个cause_event和一个effect_event。\n2.对于每一个事件，需要给出它的{Actor, Class, Action, Object (optional), Time (optional), Location (optional)}等要素，且这些要素（除了class）需要抽取原文中的片段来表示。\n3.只需要提取这些类别：{'军事行动', '外交活动', '安全事件', '政治事件', '社会事件', '科技发展', '经济事件', '航空航天活动', '装备与军备'}的事件，其他类别的事件不要提取。\n4.只需要抽取关键事件，并以重要程度排序。不关键的事件不包括在结果中。对于如何判断什么是关键事件，可以参考所给出的示例。\n5.结果要以json的格式返回。"

        for docs in selections:
            instruction_prompt = "\"Event_extraction_examples\":"
            if len(docs) < 3:
                print("some errors in sequence")
                break
            for doc in docs:
                if doc in data_dict:
                    target_doc = data_dict[doc]
                    if target_doc:
                        target_doc = {k: v for k, v in target_doc.items() if k != 'document_id'}
                        instruction_prompt += json.dumps(target_doc, ensure_ascii=False)

            instruction_prompt += "\n以上是一些事件抽取的示例，参考以上的事件抽取示例，对下面的这个文本进行因果关系事件抽取。" + same_prompt
            Alpaca = {
                "instruction": instruction_prompt,
                "input": input_prompt,
                "output": output_prompt
            }
            final_result.append(Alpaca)
    else:
        print(f"Document ID: {doc_id} not found in data.")

print(len(final_result))
# 将结果保存到文件
with open(write_filename, 'w', encoding='utf-8') as f:
    f.write('[\n')

    # 使用 tqdm 包装 enumerate(final_result)，显示进度条
    for i, entry in enumerate(tqdm(final_result, desc="Writing JSON")):
        json.dump(entry, f, ensure_ascii=False, indent=2)
        if i < len(final_result) - 1:
            f.write(',\n')

    f.write('\n]')

print("Processing complete.")

