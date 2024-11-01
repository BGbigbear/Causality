import json

def replace_keys_in_json(json_file, key_replacements):
    # 读取 JSON 文件
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 定义递归函数实现全局键替换
    def recursive_replace_keys(item):
        if isinstance(item, dict):
            new_item = {}
            for key, value in item.items():
                # 检查并替换键名
                new_key = key_replacements.get(key, key)
                new_item[new_key] = recursive_replace_keys(value)  # 递归处理值
            return new_item
        elif isinstance(item, list):
            return [recursive_replace_keys(element) for element in item]
        return item

    # 进行键替换
    data = recursive_replace_keys(data)

    # 将结果写回到 JSON 文件
    with open(json_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print("键替换已完成。")

# 使用示例
json_file = '../data/reconstruction/train2_2.json'  # JSON 文件路径
key_replacements = {
    "cause": "cause_event",
    "effect": "effect_event"
}
replace_keys_in_json(json_file, key_replacements)