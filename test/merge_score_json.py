import json
import glob
from collections import defaultdict


# 读取10个JSON文件
file_paths = glob.glob('C:/Users/thhhh/Desktop/competition/test/re/*.json')
# 输出或保存到一个新文件
output_path = "C:/Users/thhhh/Desktop/competition/test/re/merge.json"
def merge_json(file_paths,output_path):
    # 用于存储合并后的数据
    merged_data = defaultdict(lambda: {"document_id": None, "text": None, "BAD": [], "MID": []})
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                doc_id = item["document_id"]
                # 如果document_id不存在，则初始化
                if merged_data[doc_id]["document_id"] is None:
                    merged_data[doc_id]["document_id"] = doc_id
                merged_data[doc_id]["text"] = item.get("text")
                # 合并 BAD 和 MID 内容
                merged_data[doc_id]["BAD"].extend(item.get("BAD", []))
                merged_data[doc_id]["MID"].extend(item.get("MID", []))

    # 转换为列表格式
    output_data = list(merged_data.values())
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    print("合并完成，结果已保存到", output_path)

merge_json(file_paths,output_path)