def dpo_check():
    with open('../data/Alpaca/train1_dpo_bad.json', 'r', encoding='utf-8') as f:
        import json
        data = json.load(f)
        for i, d in enumerate(data):
            chosen = json.loads(d['chosen'][8:-4])
            rejected = json.loads(d['rejected'][8:-4])
            if len(chosen['causality_list']) < len(rejected['causality_list']):
                print(i)


def class_extraction():
    with open('../data/initial/train2.json', 'r', encoding='utf-8') as f:
        import json
        data = json.load(f)
        class_types = {}
        for doc in data:
            for causality in doc['causality_list']:
                cause_len = class_types.get(causality['cause']['class'], 0)
                effect_len = class_types.get(causality['effect']['class'], 0)
                class_types[causality['cause']['class']] = cause_len + 1
                class_types[causality['effect']['class']] = effect_len + 1
        print(len(class_types))
        print(class_types)


if __name__ == '__main__':
    class_extraction()
