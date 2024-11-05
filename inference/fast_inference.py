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


def chat(texts, few_shot=None, prompts=causality_prompts_v0, tokenizer=None, model=None, sam=None, inference_mode=0):
    def append_messages():
        if inference_mode == 0:
            messages[0].append({"role": "assistant", "content": chat_completion(messages[0])})
        else:
            responses = model_generation(tokenizer, model, messages, sam, use_vllm)
            for idx, response in enumerate(responses):
                messages[idx].append({"role": "assistant", "content": response})

    use_vllm = True if inference_mode == 2 else False
    messages = [[{"role": "user",
                  "content": f"{few_shot[i]}{prompts[0]}\"text\": {texts[i]}"}] for i in range(len(texts))]
    append_messages()

    for i in range(1, len(prompts)):
        if i > 1:
            for j in range(len(messages)):
                messages[j].append({"role": "user", "content": prompts[i]})

        append_messages()

    return messages


def check_json_structure(json_data):
    event_classes = ["经济事件", "科技发展", "军事行动", "安全事件", "航空航天活动", "装备与军备", "社会事件",
                     "外交活动", "政治事件"]

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
        elif idx >= start_point and rename:
            print(f"Missing document, id: {doc_id}")
        idx += 1

    return result


def preprocess(doc, causality_data, retriever, preprocess_mode):
    if preprocess_mode == 1:  # ROUGE
        shots = {"Event_extraction_examples": top_similar_text(doc['text'], causality_data, top_k=3)}
        few_shot = json.dumps(shots, ensure_ascii=False)
    else:  # preprocess_mode == 2, RAG
        vector_search_results = retriever.invoke(f"{doc['text']}")
        shots = []
        for search_result in vector_search_results:
            content = search_result.page_content.split('_', maxsplit=1)
            shot = {"text": causality_data[int(content[0])]['text'],
                    "causality_list": causality_data[int(content[0])]['causality_list']}
            shots.append(shot)
        few_shot = json.dumps(shots, ensure_ascii=False)
    return few_shot


def process_document(doc, causality_data, retriever, preprocess_mode, inference_mode, tok, model, sam, max_workers):
    if inference_mode == 0:  # API mode, doc is a dict
        few_shot = [preprocess(doc, causality_data, retriever, preprocess_mode)] if preprocess_mode != 0 else []

        max_retries, retries, circle_flag = 5, 0, False
        while retries < max_retries:
            msgs = chat([doc['text']], few_shot, global_prompts, tok, model, sam, inference_mode)
            content = msgs[0][-1]['content']
            try:
                content = content[content.find('{'): content.rfind(']') + 1] + "\n}"
                content = content.replace('\\n', '\n')
                result = json.loads(content)
                if not check_json_structure(result):
                    logging.error(f"\nJSON parsing failed on attempt {retries + 1} of document_{doc['document_id']}: "
                                  f"Json structure error.")
                    retries += 1
                    continue
                return {
                    'msg_data': {"document_id": doc['document_id'], "text": doc['text'], "analysis": msgs},
                    'result_data': {"document_id": doc['document_id'], "text": doc['text'], **result}
                }
            except ValueError as e:
                logging.error(f"\nJSON parsing failed on attempt {retries + 1} of document_{doc['document_id']}: {e}")
                retries += 1

        logging.error(f"Failed to process document {doc['document_id']} after {max_retries} attempts.")
    else:  # inference_mode == 1, 2, vllm mode, doc is a list of dict
        def get_data(data):
            return ({
                'id': data['document_id'],
                'data': {
                    "few_shot": (preprocess(data, causality_data, retriever, preprocess_mode)
                                 if preprocess_mode else ""),
                    "text": data['text']
                }
            })

        thread_data = {}
        start_time = time.time()
        print("\nFew-shot generation begin.")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(get_data, d): d for d in doc}

            for future in as_completed(futures):
                res = future.result()
                thread_data[res['id']] = res['data']

        print(f"Few-shot generation end. Total elapsed time: {convert_seconds(time.time() - start_time)}")
        few_shot, texts = [], []
        for d in doc:
            few_shot.append(thread_data[d['document_id']]['few_shot'])
            texts.append(thread_data[d['document_id']]['text'])

        msg_part, result_part = [], []  # return elements
        max_retries, retries, circle_flag = 5, 0, False
        missing_list = list(range(len(texts)))
        while retries < max_retries:
            print(f"\nTrying to generate formatted Json data, attempt {retries + 1} ...")
            msgs = chat(texts, few_shot, global_prompts, tok, model, sam, inference_mode)  # generate
            err_list, retry_texts, retry_shot = [], [], []
            for i, msg in enumerate(msgs):
                content = msg[-1]['content']
                idx = missing_list[i]  # doc id
                try:
                    content = content[content.find('{'): content.rfind(']') + 1] + "\n}"
                    content = content.replace('\\n', '\n')
                    result = json.loads(content)
                    if not check_json_structure(result):
                        logging.error(f"\nJSON parsing failed on attempt {retries + 1} "
                                      f"of document_{doc[idx]['document_id']}: Json structure error.")
                        err_list.append(idx)
                        retry_texts.append(texts[idx])
                        retry_shot.append(few_shot[idx])
                        continue
                    msg_part.append({"document_id": doc[idx]['document_id'], "text": doc[idx]['text'], "analysis": msg})
                    result_part.append({"document_id": doc[idx]['document_id'], "text": doc[idx]['text'],
                                        "causality_list": result['causality_list']})
                except ValueError as e:
                    err_list.append(idx)
                    retry_texts.append(texts[idx])
                    retry_shot.append(few_shot[idx])
                    logging.error(f"\nJSON parsing failed on attempt {retries + 1} "
                                  f"of document_{doc[idx]['document_id']}: {e}")
            if not err_list:
                break
            missing_list = err_list
            texts = retry_texts
            few_shot = retry_shot
            retries += 1
        return msg_part, result_part


def generate(start_point=0, end_point=0, preprocess_mode=0, max_workers=10, inference_mode=0, recheck=False):
    """
    Generate
    :param start_point:
    :param end_point:
    :param preprocess_mode: 0 - No preprocess; 1 - Use rouge; 2- Use rag
    :param max_workers:
    :param inference_mode: 0 - Use API; 1 - Use local model; 2 - Use vllm
    :param recheck:
    :return:
    """
    tokenizer, model, sampling_params = None, None, None
    if inference_mode != 0:
        tokenizer, model, sampling_params = load_model(True if inference_mode == 2 else False)

    with open(test_file, 'r', encoding='utf-8') as f_test:
        test_data = json.load(f_test)

    # using RAG
    causality_data, retriever = None, None
    if preprocess_mode:
        with open(causality_file, 'r', encoding='utf-8') as f_causality:
            causality_data = json.load(f_causality)
        if preprocess_mode == 2:
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
        if inference_mode == 0 or inference_mode == 1:  # API or local model
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
                        doc, causality_data, retriever, preprocess_mode,
                        inference_mode, tokenizer, model, sampling_params, max_workers
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
                                      f"Elapsed Time: {convert_seconds(total_time)}, {total_time / finished:.2f}s/it  ")
                        progress_bar(total, n, extra_info=extra_info)
                    except Exception as e:
                        print(f"Exception occurred for document {future_to_doc[future]['document_id']}: {e}")
        if inference_mode == 2:  # vllm
            docs = test_data[start_point: end_point if end_point != 0 else len(test_data)]
            if recheck:
                docs = [doc for doc in docs if doc['document_id'] in miss_set]
            if len(docs) > 0:
                msg_tmp, result_tmp = process_document(docs, causality_data, retriever, preprocess_mode,
                                                       inference_mode, tokenizer, model, sampling_params, max_workers)
                msg_data.extend(msg_tmp)
                result_data.extend(result_tmp)

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
    generate(start_point=0, end_point=10)
