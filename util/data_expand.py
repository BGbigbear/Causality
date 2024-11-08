import json


def multi_test_data(input_file, output_file, n):
    with (
        open(input_file, 'r', encoding='utf-8') as f_raw,
        open(output_file, 'w', encoding='utf-8') as f_tmp
    ):
        raw_data, tmp_data = json.load(f_raw), []
        for i in range(1, n + 1):
            for doc in raw_data:
                tmp_data.append({
                    "document_id": int(f"{100 + i}{doc['document_id']}"),
                    "text": doc['text']
                })

        json.dump(tmp_data, f_tmp, ensure_ascii=False, indent=4)


def causality_completion(raw_file, save_file):
    with (
        open(raw_file, 'r', encoding='utf-8') as f_raw,
        open(save_file, 'w', encoding='utf-8') as f_save
    ):
        data = json.load(f_raw)
        for i, doc in enumerate(data):
            causality = doc['causality_list']
            if len(causality) == 2 and causality[0]['effect'] == causality[1]['cause']:
                causality.append({
                    "causality_type": "间接",
                    "cause": causality[0]['cause'],
                    "effect": causality[1]['effect']
                })
            doc['causality_list'] = causality

        json.dump(data, f_save, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # multi_test_data('../data/initial/test2.json', '../data/initial/test2_x10.json', 10)
    causality_completion('../result/causality_test2_predict_rougeSFT_full_0.json', '../result/tmp.json')
