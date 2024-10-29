import multiprocessing as mp
import difflib
import json
import math
import time
from judge_same_event import  is_similar_event
import glob
import itertools
#一个因果对有12个事件要素，
#当因果对事件正确，得分低于3时认为是bad 0,1,2
#当因果对事件正确，得分高于6时认为是good
#mid
BAD_LIMIT = 3
GOOD_LIMIT = 10
CLASS_TYPE = ['政治事件','经济事件','安全事件','社会事件','外交活动','军事行动','科技发展','航空航天活动','装备与军备']
def convert_seconds(seconds):
    # 计算小时、分钟、秒
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    # 返回格式化的字符串
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

def read_json(file_location):
    #    从 JSON 文件读取数据并转换为字典
    if file_location ==None:
        raise ValueError("请输入文件地址")
    with open(file_location, 'r', encoding='utf-8') as file:
        data = json.load(file)
    data_dict = {event['document_id']:event for event in data}
    return data_dict


def save_to_json(data, filename="result.json"):
    """
    将数据写入 JSON 文件

    参数:
    - data: 要保存的数据（通常是一个字典或列表）
    - filename: 保存的文件名，默认为 "result.json"
    """
    try:
        with open(filename, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
        print(f"数据已成功保存到 {filename}")
    except Exception as e:
        print("保存 JSON 文件时出错:", e)

def split_dict(data_dict, num_chunks):
    # 将字典键值对拆分为相等的部分
    items = list(data_dict.items())
    chunk_size = math.ceil(len(items) / num_chunks)
    chunks = [dict(items[i:i + chunk_size]) for i in range(0, len(items), chunk_size)]
    return chunks

def longest_common_substring(s1, s2):
    if s1 is None or s2 is None:
        return "", 0  # 返回空字符串和长度为 0
    if not s1 or not s2:  # 检查字符串是否为空
        return "", 0  # 返回空字符串和长度0
    s1_size = len(s1) if s1 else 0
    s2_size = len(s2) if s2 else 0
    try:
        seq_matcher = difflib.SequenceMatcher(None, s1, s2)
        match = seq_matcher.find_longest_match(0, s1_size, 0, s2_size)
    except KeyError as e:
        print(f"KeyError occurred: {e}")
        print(s1,s2)
        return "", 0  # 返回空字符串和长度0，避免程序崩溃
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return "", 0  # 处理其他异常
    # 检查匹配长度是否大于 0
    if match.size > 0:
        return s1[match.a: match.a + match.size], match.size
    else:
        return "", 0  # 如果没有找到公共子字符串，则返回空字符串和长度为 0





#判断事件得分
#输入：两个事件 输出：得分
def judge_same_incident(pred_incident,valid_incident):
    #print(pred_incident)
    right = 0
    right_class = None
    #需要0-5中2和3的得分
    #处理单元为事件要素
    for key in pred_incident:
        #print(pred_incident[key],valid_incident[key])
        try:
            pred_len = len(pred_incident[key]) if pred_incident[key] != None else 0
            valid_len = len(valid_incident[key]) if valid_incident[key] != None else 0
            _,same_len = longest_common_substring(pred_incident[key],valid_incident[key])
        except TypeError as e:
            print(f"TypeError encountered: {e}")
            print(key,pred_incident[key],type(pred_incident[key]))
        if pred_len + valid_len != 0:
            right += 1 if same_len * 2 / (pred_len + valid_len) > 0.7 else 0
        else:
            right += 1
    if pred_incident['class'] in CLASS_TYPE:
        right_class = True
    else:
        right_class = False
    return right,right_class


def dict_values_to_string(input_dict):
    """
    将字典中的所有值连接成一个字符串。

    :param input_dict: 需要处理的字典
    :return: 字典值连接成的字符串
    """
    result = []

    for value in input_dict.values():
        # 检查值的类型并处理
        if isinstance(value, str):
            result.append(value)  # 如果值是字符串，直接添加
        elif isinstance(value, list):
            result.append(", ".join(str(item) for item in value))  # 如果值是列表，连接列表中的字符串
        elif isinstance(value, dict):
            result.append(dict_values_to_string(value))  # 如果值是字典，递归调用
        else:
            result.append(str(value))  # 其他类型转换为字符串

    return " ".join(result)  # 用空格连接所有结果

#预处理
#输入：一个完整单个text预测文本。输出：预测因果对和验证集因果对 列表
def re_casuality_list(pred_separate_dict,valid_separate_dict):
    pred_list = pred_separate_dict['causality_list']
    valid_list = valid_separate_dict['causality_list']
    pred_event = []
    valid_event = []
    for pred_pair in pred_list:
        try:
            # 尝试从字典中获取 causality_type、cause 和 effect
            causality_type = pred_pair.get('causality_type')

            cause = pred_pair['cause']
            effect = pred_pair['effect']
            pred_event.append([causality_type, cause, effect])
        except KeyError as e:
            # 捕获 KeyError 并打印出错误信息和出错的 pred_pair
            print(f"KeyError: {e} in pred_pair: {pred_pair}")
    for valid_pair in valid_list:
        valid_event.append([valid_pair['causality_type'],valid_pair['cause'],valid_pair['effect']])
    #print(pred_event,valid_event)
    return pred_event,valid_event

#一个text
#输入：预测因果对和验证集因果对 列表   输出：一个text中，所有因果对的中类。re[0]为bad，1为mid
def judge_incident_type(pred_event, valid_event):
    re = [[], []]  # 存储bad和mid分类的结果

    for pred_incident in pred_event:
        if type(pred_incident[1]) == dict:
            pred_cause_str = dict_values_to_string(pred_incident[1])
        if type(pred_incident[2]) == dict:
            pred_effect_str = dict_values_to_string(pred_incident[2])
        best_category = None  # 存储当前最佳分类（bad 或 mid）
        best_match = None  # 存储最佳分类时的valid_incident


        for valid_incident in valid_event:
            valid_cause_str = dict_values_to_string(valid_incident[1])
            valid_effect_str = dict_values_to_string(valid_incident[2])
            cause_flag = is_similar_event(s1=pred_cause_str, s2=valid_cause_str)
            effect_flag = is_similar_event(s1=pred_effect_str, s2=valid_effect_str)

            if cause_flag:
                cause_score,right_class = judge_same_incident(pred_incident[1], valid_incident[1])
                effect_score,right_class = judge_same_incident(pred_incident[2], valid_incident[2])
                total_score = cause_score + effect_score

                if not effect_flag:
                    category = "bad"
                elif total_score < BAD_LIMIT:
                    category = "bad"
                elif total_score < GOOD_LIMIT:
                    category = "mid"
                elif not right_class:
                    category = "mid"
                else:
                    category = "good"
            else:
                cause_score,right_class = judge_same_incident(pred_incident[1], valid_incident[1])
                effect_score,right_class = judge_same_incident(pred_incident[2], valid_incident[2])
                total_score = cause_score + effect_score
                category = "bad"

            # 更新最佳分类和匹配的valid_incident
            if category == 'good':
                best_category = category
                best_match = valid_incident
            elif category == 'mid' and best_category != 'good':
                best_category = category
                best_match = valid_incident
            elif category == 'bad' and best_category != 'good' and best_category != 'mid':
                best_category = category
                best_match = valid_incident

            #print(cause_flag, effect_flag, best_category, total_score)

        # 根据最佳分类更新结果
        if best_category == "bad":
            re[0].append({
                "causality_type":pred_incident[0],
                "cause":pred_incident[1],
                "effect":pred_incident[2]
            })
        elif best_category == "mid":
            re[1].append({"Pred":{
                "causality_type":pred_incident[0],
                "cause":pred_incident[1],
                "effect":pred_incident[2]
            },
            "Valid":{
                "causality_type":best_match[0],
                "cause":best_match[1],
                "effect":best_match[2]
            }})  # 将pred_incident与最佳的valid_incident一起存储
        # str1 = "".join(value for value in pred_incident[1].values() if value)
        # str2 = "".join(value for value in best_match[1].values() if value)
        # str3 = "".join(value for value in pred_incident[2].values() if value)
        # str4 = "".join(value for value in best_match[2].values() if value)
        # print(str1,'\n',str2,'\n',str3,'\n',str4)
        # print("最佳分类:", best_category)
        # print("分隔线")
    # 输出结果
    #print("Bad 分类:", re[0])
    #print("Mid 分类:", re[1])

    return re


def re_result_dict(self_data_dict, valid_data_dict):
    result_list = []

    for key, value in self_data_dict.items():
        # 获取预测和标准答案中的事件
        self_event = value
        if key in valid_data_dict:
            valid_event = valid_data_dict[key]
        else:
            raise ValueError(f"document_id: {key} 未在标准答案中找到对应事件")

        # 处理事件
        pred_event, valid_event = re_casuality_list(self_event, valid_event)
        re = judge_incident_type(pred_event, valid_event)

        # 构造结果字典
        result_dict = {
            'document_id': key,
            'text': value['text'],
            "BAD":re[0],
            "MID":re[1]
        }

        # 将当前字典的副本添加到列表中
        result_list.append(result_dict.copy())

    #print(result_list)
    return result_list


if __name__ == '__main__':
    #pred_location = 'C:/Users/thhhh/Desktop/competition/test/test.json'
    valid_location = 'C:/Users/thhhh/Desktop/competition/test/train1.json'
    pred_file_paths = glob.glob('C:/Users/thhhh/Desktop/competition/test/pred/*.json')
    start = time.time()
    i = 0
    for pred_location in pred_file_paths:
        pred_data_dict = read_json(pred_location)
        valid_data_dict = read_json(valid_location)
        num_cores = 14
        print(f'num_cores:{num_cores}')
        splited_pred_data_dict_list = split_dict(pred_data_dict, num_cores)

        with mp.Pool(num_cores) as pool:
            # 并行处理每个子字典
            results = pool.starmap(re_result_dict, [(chunk, valid_data_dict) for chunk in splited_pred_data_dict_list])
        #print(i,'\n',results)
        result = list(itertools.chain.from_iterable(results))
        save_to_json(result,f'C:/Users/thhhh/Desktop/competition/test/re/re{i}.json')
        pool.close()
        pool.join()
        i += 1

    end_ = time.time()
    print('运行时间', convert_seconds(end_-start))