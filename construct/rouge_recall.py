import json
import random
from collections import defaultdict
from typing_extensions import Counter
from tqdm import tqdm  # 引入tqdm库

from util.rouge import ngram_rouge

def final_sel(open_filename):
# 从文件中加载数据
    with open(open_filename, "r", encoding="utf-8") as f:
       data = json.load(f)



    # 存储每个文档的相似度得分
    results = {}
    # 存储最终选择的文档
    final_selections = {}

    # 添加进度条，遍历每一个文档，将其作为候选文本
    for candidate_doc in tqdm(data, desc="Processing documents"):
        candidate_id = candidate_doc.get("document_id")
        candidate_text = candidate_doc.get("text", "")

        # 创建一个字典用于存储当前文档与其他文档的相似度得分
        scores = {}

        # 遍历其他文档，计算与候选文档的 Rouge-n 得分
        for reference_doc in data:
            reference_id = reference_doc.get("document_id")
            reference_text = reference_doc.get("text", "")

            # 不与自己比较
            if candidate_id != reference_id:
                score = ngram_rouge(candidate_text, reference_text, 3)
                scores[reference_id] = score

        # 对当前候选文档的相似文档按得分降序排序，并获取最相似的前5个文档
        top_5_similar_docs = [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]]

        # 将结果存储
        results[candidate_id] = top_5_similar_docs

    # 添加进度条，遍历每个文档的相似文档
    for doc_id, similar_docs in tqdm(results.items(), desc="Selecting similar documents"):
        # 用于存储每次选择的结果
        selections = []

        # 进行3次选择
        for _ in range(3):
            while True:
                # 随机选择3个文档
                chosen_docs = random.sample(similar_docs, 3)

                # 确保当前选择与之前的选择不完全相同
                if not any(set(chosen_docs) == set(previous) for previous in selections):
                    selections.append(chosen_docs)
                    break

        # 将选出的文档组合成最终结果，转化为单层列表
        final_selections[doc_id] = selections

    return final_selections
