import json
import logging
import sys
import threading
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.configuration import *
from inference.inference_tools import chat_completion, load_model, model_generation
# from util.rag import load_retriever
from util.rouge import top_similar_text

logging.disable(logging.WARNING)
logging.basicConfig(
    level=logging.WARNING,  # 设置日志级别
    # format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 日志格式
    # datefmt='%Y-%m-%d %H:%M:%S',  # 日期格式
    # filename='app.log',  # 日志输出到文件
    # filemode='w'  # 写模式，'a'为追加模式
)

progress_lock = threading.Lock()


def progress_bar(i, n, length=50, extra_info=""):
    with progress_lock:
        percent = i / n
        # Calculate the number of '=' to display in the progress bar
        filled_length = int(length * percent)
        # Create the progress bar string
        bar = '=' * filled_length + '-' * (length - filled_length)
        # Print the progress bar with the percentage
        sys.stdout.write(f'\r[{bar}] {percent:.2%} {extra_info}')
        sys.stdout.flush()


def convert_seconds(seconds):
    if seconds < 60:
        return f"{seconds:.0f} s"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{minutes:.0f} m {remaining_seconds:.0f} s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        remaining_seconds = seconds % 60
        return f"{hours:.0f} h {minutes:.0f} m {remaining_seconds:.0f} s"


def chat(text, few_shot=None, prompts=causality_prompts_v0, tokenizer=None, model=None, using_api=False):
    def append_messages():
        if not using_api:
            messages.append({"role": "assistant", "content": model_generation(tokenizer, model, messages)})
        else:
            messages.append({"role": "assistant", "content": chat_completion(messages)})

    # q1 = f"文本：{text}" + f"\n参考：\n{few_shot}" if few_shot else f"文本：{text}"
    # q1 += f"\n({prompts[1]})"
    #
    # messages = [
    #     {"role": "system", "content": prompts[0]},
    #     {"role": "user", "content": q1}
    # ]

    messages = [
        {"role": "user", "content": f"{few_shot}{prompts[0]}\"text\": {text}"}
    ]
    append_messages()

    for i in range(1, len(prompts)):
        if i > 1:
            messages.append({"role": "user", "content": prompts[i]})

        append_messages()

    return messages


def check_json_structure(json_data):
    event_classes = ["经济事件", "科技发展", "军事行动", "安全事件", "航空航天活动", "装备与军备", "社会事件", "外交活动", "政治事件"]

    if "causality_list" not in json_data:
        print(f"\nDocument is missing 'causality_list'.")
        return False

    for idx, causality in enumerate(json_data["causality_list"]):
        if "causality_description" not in causality:
            print(f"\nDocument is missing 'causality_description'.")
            return False
        if "causality_type" not in causality:
            print(f"\nDocument is missing 'causality_type'.")
            return False

        for role in event_roles:
            if not isinstance(causality, dict):
                print(f"\nEvent {idx}, '{role}' is not dict")
                return False

            event = causality.get(role, {})

            missing_keys = required_keys - event.keys()
            extra_keys = event.keys() - required_keys

            if missing_keys:
                print(f"\nEvent {idx}, '{role}' is missing keys: {missing_keys}.")
                return False
            if extra_keys:
                print(f"\nEvent {idx}, '{role}' has extra keys: {extra_keys}.")
                return False
            if event['class'] not in event_classes:
                print(f"\nEvent {idx}, '{role}' type error: {event['class']}.")
            #     return False

    return True


def rearrange(test_data, result_data, start_point, end_point, rename=True):
    pred_data = {str(pred['document_id']): pred for pred in result_data}

    result = []
    idx = 0
    while idx < end_point:
        doc_id = test_data[idx]['document_id']
        data = pred_data.get(str(doc_id), None)
        if data:
            if rename:
                tmp_data = json.dumps(data, ensure_ascii=False)  # transfer into string for replacement
                tmp_data = tmp_data.replace("cause_event", "cause").replace("effect_event", "effect")
                data = json.loads(tmp_data)
            result.append(data)
        elif idx >= start_point:
            print(f"Missing document, id: {doc_id}")
        idx += 1

    return result


def process_document(doc, causality_data, retriever, rouge, rag, prompts, using_api, tokenizer, model):
    few_shot = ""
    if rouge:
        shots = {"Event_extraction_examples": top_similar_text(doc['text'], causality_data, top_k=3)}
        few_shot += json.dumps(shots, ensure_ascii=False)
    elif rag:
        vector_search_results = retriever.invoke(f"{doc['text']}")
        shots = []
        for search_result in vector_search_results:
            content = search_result.page_content.split('_', maxsplit=1)
            shot = {"text": causality_data[int(content[0])]['text'],
                    "causality_list": causality_data[int(content[0])]['causality_list']}
            shots.append(shot)
        few_shot += json.dumps(shots, ensure_ascii=False)

    msgs = chat(doc['text'], few_shot, prompts, tokenizer, model, using_api)  # generate
    content = msgs[-1]['content']

    max_retries, retries, circle_flag = 5, 0, False
    while retries < max_retries:
        if circle_flag:
            # completion = chat_completion(msgs[:-1])
            # content = str(completion.choices[0].message.content)
            msgs = chat(doc['text'], few_shot, prompts, tokenizer, model, using_api)
            content = msgs[-1]['content']
        try:
            result = json.loads(content[content.find('{'): content.rfind(']') + 1] + "\n}")
            if not check_json_structure(result):
                logging.error(f"\nJSON parsing failed on attempt {retries + 1} of document_{doc['document_id']}: "
                              f"Json structure error.")
                circle_flag = True
                retries += 1
                continue
            return {
                'msg_data': {"document_id": doc['document_id'], "text": doc['text'], "analysis": msgs},
                'result_data': {"document_id": doc['document_id'], "text": doc['text'], **result}
            }
        except ValueError as e:
            logging.error(f"\nJSON parsing failed on attempt {retries + 1} of document_{doc['document_id']}: {e}")
            circle_flag = True
            retries += 1

    logging.error(f"Failed to process document {doc['document_id']} after {max_retries} attempts.")


def generate(start_point=0, end_point=0, rouge=False, rag=False, max_workers=10, using_api=False, recheck=False):
    tokenizer, model = None, None
    if not using_api:
        tokenizer, model = load_model(model_path)

    with open(test_file, 'r', encoding='utf-8') as f_test:
        test_data = json.load(f_test)

    # using RAG
    causality_data, retriever = None, None
    if rag or rouge:
        with open(causality_file, 'r', encoding='utf-8') as f_causality:
            causality_data = json.load(f_causality)
        if rag:
            # retriever = load_retriever(causality_data, db_path)
            pass

    start_time = time.time()
    n = len(test_data) if start_point == end_point == 0 \
        else (len(test_data) - start_point if end_point == 0 else end_point - start_point)  # total task number
    msg_data, result_data = [], []

    if recheck or (start_point != 0 and os.path.exists(analysis_file) and os.path.exists(pred_file)):
        with (
            open(analysis_file, 'r', encoding='utf-8') as f_analysis,
            open(pred_file, 'r', encoding='utf-8') as f_pred
        ):
            msg_data, result_data = json.load(f_analysis), json.load(f_pred)
            n += start_point if not recheck else 0  # recheck or resume

    progress_bar(0, n, extra_info="Initializing...")

    init_finished = len(msg_data)
    miss_set = None
    if recheck:
        doc_set = {test['document_id'] for test in test_data}
        cur_set = {result['document_id'] for result in result_data}
        miss_set = doc_set - cur_set
        print(f"\nMissing document set: {miss_set}")
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_doc = {}
            for i, doc in enumerate(test_data):
                if start_point != 0 and i < start_point:
                    # progress_bar(i - start_point + 1, n)
                    continue
                if end_point != 0 and i >= end_point:
                    break
                if recheck and doc['document_id'] not in miss_set:
                    continue

                future = executor.submit(
                    process_document,
                    doc, causality_data, retriever, rouge, rag, global_prompts, using_api, tokenizer, model
                )
                future_to_doc[future] = doc

            for future in as_completed(future_to_doc):
                try:
                    res = future.result()
                    if res:
                        msg_data.append(res['msg_data'])
                        result_data.append(res['result_data'])

                    total = len(msg_data)
                    finished = max(total - init_finished, 1)
                    total_time = time.time() - start_time
                    extra_info = (f"Task {total}/{n} | "
                                  f"Elapsed Time: {convert_seconds(total_time)}, {total_time/finished:.2f}s/it   ")
                    progress_bar(total, n, extra_info=extra_info)
                except Exception as e:
                    print(f"Exception occurred for document {future_to_doc[future]['document_id']}: {e}")
        print("\nInference finished")
    finally:
        print("Start writing files")
        msg_data = rearrange(test_data, msg_data, start_point, len(test_data) if end_point == 0 else end_point, False)
        result_data = rearrange(test_data, result_data, start_point, len(test_data) if end_point == 0 else end_point)
        with (
            open(analysis_file, 'w', encoding='utf-8') as f_analysis,
            open(pred_file, 'w', encoding='utf-8') as f_pred
        ):
            json.dump(msg_data, f_analysis, ensure_ascii=False, indent=4)
            json.dump(result_data, f_pred, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    generate(start_point=0, end_point=10, rouge=True)
