import multiprocessing as mp
import itertools
import json
import math
import time


def split_dict(data_dict, num_chunks):
    # 将字典键值对拆分为相等的部分
    items = list(data_dict.items())
    chunk_size = math.ceil(len(items) / num_chunks)
    chunks = [dict(items[i:i + chunk_size]) for i in range(0, len(items), chunk_size)]
    return chunks


def llcs(X, Y):
    m, n = len(X), len(Y)
    # 初始化 DP 数组
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_len = 0  # 用于记录最长公共子串长度

    # 填充 DP 表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                max_len = max(max_len, dp[i][j])
            else:
                dp[i][j] = 0

    return max_len


def read_json(file_location):
    #    从 JSON 文件读取数据并转换为字典
    if file_location is None:
        raise ValueError("请输入文件地址")
    with open(file_location, 'r', encoding='utf-8') as file:
        data = json.load(file)
    data_dict = {event['document_id']: event for event in data}
    # 输出字典内容
    # print(data_dict)
    return data_dict


# 一个text
def cal_causality_list(self_causality_list, valid_causality_list):
    """
    self_causality_list格式
    [
    {
    'causality_description': '提坦公司赢得价值4500万美元的合同直接导致提坦公司为MSA计划提供系统工程和计划管理支持',
    'causality_type': '直接',
    'cause': {'event_description': '提坦公司赢得一份价值4500万美元的合同', 'actor': '提坦公司', 'class': '经济事件', 'action': '赢得', 'time': '', 'location': '', 'object': '一份价值4500万美元的合同'},
    'effect': {'event_description': '提坦公司为MSA计划提供系统工程和计划管理支持', 'actor': '提坦公司', 'class': '科技发展', 'action': '提供', 'time': '', 'location': '', 'object': '系统工程和计划管理支持'}
    },
    {'causality_description': '提坦公司为MSA计划提供系统工程和计划管理支持直接导致提坦公司需要发展特殊系统结构', 'causality_type': '直接', 'cause': {'event_description': '提坦公司为MSA计划提供系统工程和计划管理支持', 'actor': '提坦公司', 'class': '科技发展', 'action': '提供', 'time': '', 'location': '', 'object': '系统工程和计划管理支持'}, 'effect': {'event_description': '提坦公司需要发展特殊系统结构', 'actor': '提坦公司', 'class': '科技发展', 'action': '需要发展', 'time': '', 'location': '', 'object': '特殊系统结构'}}, {'causality_description': '提坦公司需要发展特殊系统结构直接导致这种系统结构的开发将用于改进P-3C飞机的性能', 'causality_type': '直接', 'cause': {'event_description': '提坦公司需要发展特殊系统结构', 'actor': '提坦公司', 'class': '科技发展', 'action': '需要发展', 'time': '', 'location': '', 'object': '特殊系统结构'}, 'effect': {'event_description': '这种系统结构的开发将用于改进P-3C飞机的性能', 'actor': '这种系统结构的开发', 'class': '科技发展', 'action': '用于改进', 'time': '', 'location': '', 'object': 'P-3C飞机的性能'}}
    ]

    :param self_causality_list: 预测列表    格式是列表
    :param valid_causality_list: 验证列表
    :return: max_right,
    max_actor,max_class,max_action,max_time,max_location,max_object, max_text, max_type
    第一个返回是最终得分，其他可以不要
    """
    max_right = 0

    causality_list = self_causality_list if len(self_causality_list) > len(
        valid_causality_list) else valid_causality_list
    if causality_list == self_causality_list:
        flag = True
    else:
        flag = False
    for p in generate_permutations(causality_list):
        if flag:
            right_min = acquire_min_score(p, valid_causality_list)
        else:
            right_min = acquire_min_score(p, self_causality_list)
        if right_min > max_right:
            max_right = right_min
    return max_right


# self_list 为一个完整causality_list
def acquire_min_score(self_list, valid_list):
    right = 0
    n = min(len(self_list), len(valid_list))
    for i in range(n):
        # 因果对，
        sepair_list = []
        vapair_list = []
        self = self_list[i]
        valid = valid_list[i]
        if self['causality_type'] == valid['causality_type']:
            right += 1
        sepair_list.extend(list(self['cause'].values()))
        sepair_list.extend(list(self['effect'].values()))
        vapair_list.extend(list(valid['cause'].values()))
        vapair_list.extend(list(valid['effect'].values()))
        # print(sepair_list,vapair_list)
        for j in range(12):
            if vapair_list[j] is None or sepair_list[j] is None:
                continue
            else:
                valid_num = len(vapair_list[j])
                self_num = len(sepair_list[j])
                same_num = llcs(sepair_list[j], vapair_list[j])
            if self_num + valid_num == 0 or same_num * 2 / (self_num + valid_num) > 0.7:
                right += 1  # 只在这一处累加 right
    return right


def generate_permutations(arr):
    for perm in itertools.permutations(arr):
        yield perm  # 使用生成器逐个返回排列


def re_result_dict(self_data_dict, valid_data_dict):
    # 占用内存
    # rigth_num,Pdeno,Rdeno = 0,0,0
    result_dict = {}
    for key, value in self_data_dict.items():
        # 获得预测和标答地text
        self_event = value
        if key in valid_data_dict:
            # 一次查找
            valid_event = valid_data_dict[key]
        else:
            raise ValueError("document_id:", key, "未在标准答案中找到对应事件")
        # 事件对列表
        self_casuality_list = self_event["causality_list"]
        Pdeno = len(self_casuality_list) * 13  # P的分母，原因要素和结果要素一共有12个
        valid_casuality_list = valid_event['causality_list']
        Rdeno = len(valid_casuality_list) * 13  # R的分母
        # se为一个完整text
        min_score = cal_causality_list(self_casuality_list, valid_casuality_list)
        result_dict[key] = [min_score, Pdeno, Rdeno]

    # print(result_dict)
    return result_dict


def re_PandR(result_dict):
    sum_numerator = 0
    sum_Pdeno = 0
    sum_Rdeno = 0
    for value in result_dict.values():
        sum_numerator += value[0]  # 将每个列表的第一个元素相加
        sum_Pdeno += value[1]
        sum_Rdeno += value[2]
    return sum_numerator, sum_Pdeno, sum_Rdeno


def re_F1(results):
    (max_sum_numerator, max_sum_Pdeno, max_sum_Rdeno,) = 0, 0, 0
    for i in range(len(results)):
        sum_numerator, sum_Pdeno, sum_Rdeno = re_PandR(results[i])
        max_sum_numerator += sum_numerator
        max_sum_Pdeno += sum_Pdeno
        max_sum_Rdeno += sum_Rdeno
    print('预测正确', max_sum_numerator, '预测量', max_sum_Pdeno, '验证集量', max_sum_Rdeno)
    P = max_sum_numerator / max_sum_Pdeno
    R = max_sum_numerator / max_sum_Rdeno
    F1 = 2 * P * R / (P + R)
    print('  p:', P, '  r:', R, '  f1:', F1)
    return F1


# list1 = [
#             {
#                 "causality_description": "巴德克核物理研究所希望提升火箭发动机的效率导致物理学家开始运行名为SMOLA的装置",
#                 "causality_type": "直接",
#                 "cause": {
#                     "event_description": "巴德克核物理研究所希望提升火箭发动机的效率",
#                     "actor": "巴德克核物理研究所",
#                     "class": "科技发展",
#                     "action": "希望提升",
#                     "time": "",
#                     "location": "",
#                     "object": "火箭发动机的效率"
#                 },
#                 "effect": {
#                     "event_description": "物理学家开始运行名为SMOLA的装置",
#                     "actor": "物理学家",
#                     "class": "科技发展",
#                     "action": "开始运行",
#                     "time": "",
#                     "location": "",
#                     "object": "名为SMOLA的装置"
#                 }
#             }
# ]
# list2 = [
#             {
#                 "causality_description": "巴德克核物理研究所希望提升火箭发动机的效率导致物理学家开始运行名为SMOLA的装置",
#                 "causality_type": "直接",
#                 "cause": {
#                     "event_description": "巴德克核物理研究所希望提升火箭发动机的效率",
#                     "actor": "巴德克核物理研究所",
#                     "class": "科技发展",
#                     "action": "希望提升",
#                     "time": "",
#                     "location": "",
#                     "object": "火箭发动机的效率"
#                 },
#                 "effect": {
#                     "event_description": "物理学家开始运行名为SMOLA的装置",
#                     "actor": "物理学家",
#                     "class": "科技发展",
#                     "action": "开始运行",
#                     "time": "",
#                     "location": "",
#                     "object": "名为SMOLA的装置"
#                 }
#             }
# ]
# print(cal_causality_list(list1,list2))
if __name__ == '__main__':
    self_location = 'C:\\Users\\thhhh\\Desktop\\competition\\test\\train2\\pred/causality_train2cot_predict_rouge_full_0.json'
    valid_location = 'C:\\Users\\thhhh\\Desktop\\competition\\test\\train2\\train2/train2.json'

    start = time.time()

    self_data_dict = read_json(self_location)
    valid_data_dict = read_json(valid_location)
    num_cores = 1
    print(f'num_cores:{num_cores}')
    splited_self_data_dict_list = split_dict(self_data_dict, num_cores)

    # 创建进程池
    with mp.Pool(num_cores) as pool:
        # 并行处理每个子字典
        results = pool.starmap(re_result_dict, [(chunk, valid_data_dict) for chunk in splited_self_data_dict_list])

    pool.close()
    pool.join()

    re_F1(results)
    end_ = time.time()
