import heapq
from collections import Counter


def ngram_rouge(candidate, reference, n):
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
        score = ngram_rouge(doc['text'], text, n)
        if len(outputs) < top_k:
            heapq.heappush(outputs, (score, i))
        elif score > outputs[0][0]:
            heapq.heappushpop(outputs, (score, i))

    return [{'text': causality_data[idx[1]]['text'], 'causality_list': causality_data[idx[1]]['causality_list']}
            for idx in sorted(outputs, key=lambda x: x[0], reverse=True)]
