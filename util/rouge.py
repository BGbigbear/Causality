import heapq
from collections import Counter


def ngram_rouge(candidate, reference, n=2):
    candidate_ngrams = [candidate[i:i + n] for i in range(len(candidate) - n + 1)]
    reference_ngrams = [reference[i:i + n] for i in range(len(reference) - n + 1)]

    candidate_counts = Counter(candidate_ngrams)
    reference_counts = Counter(reference_ngrams)

    overlap = sum((candidate_counts & reference_counts).values())
    total_ngrams = sum(reference_counts.values())

    if total_ngrams == 0:
        return 0.0

    return overlap / total_ngrams


def top_similar_text(text, causality_data, top_k=5, n=2):
    pass
    outputs = []
    for i, doc in enumerate(causality_data):
        if doc['text'] == text:  # skip same text
            continue
        score = ngram_rouge(doc['text'], text, n)
        if len(outputs) < top_k:
            heapq.heappush(outputs, (score, i))
        elif score > outputs[0][0]:
            heapq.heappushpop(outputs, (score, i))

    return [{'text': causality_data[idx[1]]['text'], 'causality_list': causality_data[idx[1]]['causality_list']}
            for idx in sorted(outputs, key=lambda x: x[0], reverse=True)]


def select_best_output(response, n, idx):
    # 提取所有输出文本
    outputs_texts = [f"{output.text!r}" for output in response.outputs[:n]]

    # 存储每个输出的平均分
    average_scores = []

    # 遍历每个输出，并计算其与其他输出的 ROUGE 分数
    for i in range(n):
        scores = []
        for j in range(n):
            if i != j:
                # 计算当前输出与其他输出的 ROUGE 分数
                score = ngram_rouge(outputs_texts[i], outputs_texts[j])
                scores.append(score)

        # 计算当前输出的平均分
        average_score = sum(scores) / len(scores) if scores else 0
        average_scores.append(average_score)

    sorted_indices = sorted(range(len(average_scores)), key=lambda x: average_scores[x], reverse=True)

    # 获取平均分最高的输出索引
    best_index = sorted_indices[idx-1]

    # 返回平均分最高的输出文本
    return outputs_texts[best_index]
