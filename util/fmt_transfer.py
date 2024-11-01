import json


def alpaca2sharegpt(alpaca_path, sharegpt_path):
    with (
        open(alpaca_path, 'r', encoding='utf-8') as f_alpaca,
        open(sharegpt_path, 'w', encoding='utf-8') as f_sharegpt
    ):
        alpaca_data, sharegpt_data = json.load(f_alpaca), []
        for a_data in alpaca_data:
            sharegpt_data.append({
                "conversations": [
                    {
                        "from": "human",
                        "value": a_data['instruction'] + a_data['input']
                    },
                    {
                        "from": "gpt",
                        "value": a_data['output']
                    }
                ],
                "kto_tag": a_data['kto_tag']
            })
        json.dump(sharegpt_data, f_sharegpt, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    alpaca2sharegpt('../data/Alpaca/train1_kto.json', '../data/Alpaca/train1_kto_sharegpt.json')
