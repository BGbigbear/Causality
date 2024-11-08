import json


def dpo_check():
    with open('../data/Alpaca/train2_dpo_mid.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        for i, d in enumerate(data):
            chosen = json.loads(d['chosen'][8:-4])
            rejected = json.loads(d['rejected'][8:-4])
            if len(chosen['causality_list']) != len(rejected['causality_list']):
                print(i)


def number_of_causality():
    # file = "../result/causality_test2_predict_rougeSFT_full_0.json"
    # file = "../data/reconstruction/train2_e.json"
    file = "../result/causality_test2_predict_rougeSFTsc_full_3a.json"
    # file = '../result/tmp.json'
    number = {}
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for i, d in enumerate(data):
            length = len(d['causality_list'])
            number[length] = number.get(length, 0) + 1

        print(number)


def class_extraction():
    file = '../data/initial/train2.json'
    with open(file, 'r', encoding='utf-8') as f:
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


def extract_indirect():
    file = '../result/causality_test2_predict_rougeSFT_full_0.json'
    # file = "../result/causality_test2_predict_rougeSFTsc_full_3.json"
    # file = '../data/initial/train2.json'
    # file = '../result/causality_test2_predict_rougeSFT_full_0c.json'
    cnt = 0
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for i, doc in enumerate(data):
            causality = doc['causality_list']
            if len(causality) == 2 and causality[0]['effect'] == causality[1]['cause']:
                # if len(causality) == 1 and causality[0]['causality_type'] == '间接':
                print(i, doc['document_id'])
                cnt += 1
    print(cnt)


def check_integrity():
    # file = '../result/causality_test2_predict_rougeSFT_full_0.json'
    file = '../data/initial/train2.json'
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for i, doc in enumerate(data):
            causality_list = doc['causality_list']
            for causality in causality_list:
                for role in ['cause', 'effect']:
                    for k, v in causality[role].items():
                        if k in ['actor'] and '，' in v:
                            print(i, doc['document_id'])
                            break


def compare_number():
    file1 = '../result/causality_test2_predict_rougeSFT_full_0.json'
    file2 = '../result/causality_test2_predict_rougeSFTsc_full_3.json'
    with (
        open(file1, 'r', encoding='utf-8') as f1,
        open(file2, 'r', encoding='utf-8') as f2
    ):
        data1, data2 = json.load(f1), json.load(f2)
        cnt = 0
        for doc1, doc2 in zip(data1, data2):
            # if len(doc1['causality_list']) > len(doc2['causality_list']):
            if (len(doc1['causality_list']) == 2 and
                    doc1['causality_list'][0]['effect'] == doc1['causality_list'][1]['cause'] and
                    len(doc2['causality_list']) >= 3):
                print(doc1['document_id'])
                cnt += 1
        print(cnt)


def find_same():
    file1 = '../result/causality_test2_predict_rougeSFT_full_0a.json'
    # file1 = '../result/causality_test2_predict_rougeSFTsc_full_3a.json'
    file2 = '../data/initial/train2.json'
    with (
        open(file1, 'r', encoding='utf-8') as f1,
        open(file2, 'r', encoding='utf-8') as f2
    ):
        data1, data2 = json.load(f1), json.load(f2)
        f2.seek(0)
        raw = f2.read()
        cnt = 0
        for i, doc1 in enumerate(data1):
            if doc1['text'] in raw:
                for doc2 in data2:
                    if doc1['text'] == doc2['text'] and doc1['causality_list'] == doc2['causality_list']:
                        print(i, doc1['document_id'])
                        # data1[i]['causality_list'] = doc2['causality_list']
                        cnt += 1
        print(cnt)

    # with open(file1, 'w', encoding='utf-8') as f:
    #     json.dump(data1, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # extract_indirect()
    # number_of_causality()
    # check_integrity()
    # compare_number()
    find_same()
