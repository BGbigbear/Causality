
import multiprocessing as mp
import difflib
import itertools
import json
import math
import time

def convert_seconds(seconds):
    # 计算小时、分钟、秒
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    # 返回格式化的字符串
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
def split_dict(data_dict, num_chunks):
    # 将字典键值对拆分为相等的部分
    items = list(data_dict.items())
    chunk_size = math.ceil(len(items) / num_chunks)
    chunks = [dict(items[i:i + chunk_size]) for i in range(0, len(items), chunk_size)]
    return chunks

def read_json(file_location):
    #    从 JSON 文件读取数据并转换为字典
    if file_location ==None:
        raise ValueError("请输入文件地址")
    with open(file_location, 'r', encoding='utf-8') as file:
        data = json.load(file)
    data_dict = {event['document_id']:event for event in data}
    # 输出字典内容
    #print(data_dict)
    return data_dict

#一个text
def re_P_R(self_casuality_list,valid_casuality_list):
    va_event = []
    se_event = []
    for self_casuality in self_casuality_list:
        reason = self_casuality['cause']
        result = self_casuality['effect']
        reason_list = list(reason.values())
        result_list = list(result.values())
        se_event.append(reason_list)
        se_event.append(result_list)
    #print(se_event)
    for valid_casuality in valid_casuality_list:
        reason = valid_casuality['cause']
        result = valid_casuality['effect']
        reason_list = list(reason.values())
        result_list = list(result.values())
        va_event.append(reason_list)
        va_event.append(result_list)

    #print(va_event)
    max_right = 0
    max_actor = 0
    max_class = 0
    max_time = 0
    max_location = 0
    max_action = 0
    max_object = 0
    max_text = ''


    #se_event为多个事件

    event = se_event if len(se_event)>len(va_event) else va_event
    if event == se_event:
        flag = True
    else:
        flag = False
    #p为一个text
    for p in generate_permutations(event):

        if flag:
            right_min,actor_min,class_min, action_min,time_min,location_min, object_min = acquire_min_score(p, va_event)

        else:
            right_min,actor_min,class_min, action_min,time_min,location_min, object_min = acquire_min_score(p, se_event)

        if right_min > max_right:
            #rr = num
            max_right = right_min
            max_time = time_min
            max_actor = actor_min
            max_class = class_min
            max_location = location_min
            max_text = p
            max_action = action_min
            max_object = object_min
        #print(max)
    #print(rr,max_right,max_action,max_object, max_text)
    return max_right,max_actor,max_class,max_action,max_time,max_location,max_object, max_text



#self_list 为一个完整text中，事件列表
def acquire_min_score(self_list ,valid_list):
    right = 0
    action_num = 0
    object_num = 0
    class_num = 0
    time_num = 0
    location_num = 0
    actor_num = 0
    n = min(len(self_list),len(valid_list))
    for i in range(n):
        #为事件
        self = self_list[i]
        valid = valid_list[i]
        #需要0-5中2和3的得分
        #处理单元为事件要素
        for j in range(len(self)):
            if valid[j] == None:
                valid_num = 0
            else:
                valid_num = len(valid[j])
            if self[j] == None:
                self_num = 0
            else:
                self_num = len(self[j])
            if valid[j] != None and self[j] != None:
                _,same_num = longest_common_substring(self[j],valid[j])
            else:
                same_num = 0
            if self_num + valid_num == 0 or same_num * 2 / (self_num + valid_num) > 0.7:
                right += 1  # 只在这一处累加 right
                if j == 2:
                    action_num += 1
                elif j == 5:
                    object_num += 1
                elif j == 0:
                    actor_num += 1
                elif j == 4:
                    location_num += 1
                elif j == 3:
                    time_num += 1
                elif j == 1:
                    class_num += 1
    return right,actor_num,class_num, action_num, time_num,location_num,object_num


def longest_common_substring(s1, s2):
    seq_matcher = difflib.SequenceMatcher(None, s1, s2)
    s1_size = len(s1) if s1 != None else 0
    s2_size = len(s2) if s2 != None else 0
    match = seq_matcher.find_longest_match(0, s1_size, 0, s2_size)
    return s1[match.a: match.a + match.size], match.size


def generate_permutations(arr):
    for perm in itertools.permutations(arr):
        yield perm  # 使用生成器逐个返回排列


def re_result_dict(self_data_dict,valid_data_dict):
    number=0
    #占用内存
    # rigth_num,Pdeno,Rdeno = 0,0,0
    result_dict = {}
    for key,value in self_data_dict.items():
        #获得预测和标答地text
        self_event = value
        if key in valid_data_dict:
            #一次查找
            valid_event = valid_data_dict[key]
        else:
            raise ValueError("document_id:",key,"未在标准答案中找到对应事件")
        # 事件对列表
        self_casuality_list = self_event["causality_list"]
        Pdeno = len(self_casuality_list) * 12  # P的分母，原因要素和结果要素一共有12个
        valid_casuality_list = valid_event['causality_list']
        Rdeno = len(valid_casuality_list) * 12  # R的分母
        #se为一个完整text
        min_score,actor_,class_,action_,time_,location_,object_,min_event = re_P_R(self_casuality_list,valid_casuality_list)
        result_dict[key] = [min_score,Pdeno,Rdeno,actor_,class_,action_,time_,location_,object_]
        number+=1

    #print(result_dict)
    return result_dict


def re_PandR(result_dict):
    sum_numerator = 0
    sum_Pdeno = 0
    sum_Rdeno = 0
    sum_action = 0
    sum_object = 0
    sum_actor = 0
    sum_class = 0
    sum_time = 0
    sum_location = 0
    for value in result_dict.values():
        sum_numerator += value[0]  # 将每个列表的第一个元素相加
        sum_Pdeno += value[1]
        sum_Rdeno += value[2]
        sum_actor += value[3]
        sum_class += value[4]
        sum_action += value[5]
        sum_time += value[6]
        sum_location += value[7]
        sum_object += value[8]
    return sum_numerator,sum_Pdeno,sum_Rdeno,sum_actor,sum_class,sum_action,sum_time,sum_location,sum_object

def re_F1(results):
    (max_sum_numerator,max_sum_Pdeno,max_sum_Rdeno,
     max_sum_actor,max_sum_class,max_sum_action,max_sum_time,max_sum_location,max_sum_object) = 0, 0,0, 0, 0,0,0,0,0
    for i in range(len(results)):
        sum_numerator,sum_Pdeno,sum_Rdeno,sum_actor,sum_class,sum_action,sum_time,sum_location,sum_object = re_PandR(results[i])
        #print('预测正确', sum_numerator, '预测量', sum_Pdeno, '标答量', sum_Rdeno, 'action', sum_action, 'object',
        #      sum_object,'actor',sum_actor,'class',sum_class,'time',sum_time,'location',sum_location)
        max_sum_numerator += sum_numerator
        max_sum_Pdeno += sum_Pdeno
        max_sum_Rdeno += sum_Rdeno
        max_sum_actor += sum_actor
        max_sum_class+= sum_class
        max_sum_action += sum_action
        max_sum_time+= sum_time
        max_sum_location+= sum_location
        max_sum_object += sum_object
    print('预测正确', max_sum_numerator, '预测量', max_sum_Pdeno, '标答量', max_sum_Rdeno, 'action', max_sum_action, 'object',
              max_sum_object,'actor',max_sum_actor,'class',max_sum_class,'time',max_sum_time,'location',max_sum_location)
    actor = round(max_sum_actor/max_sum_numerator,6)
    classs = round(max_sum_class/max_sum_numerator,6)
    action = round(max_sum_action/max_sum_numerator,6)
    timee = round(max_sum_time/max_sum_numerator,6)
    location = round(max_sum_location/max_sum_numerator,6)
    object = round(max_sum_object/max_sum_numerator,6)
    P = max_sum_numerator / max_sum_Pdeno
    R = max_sum_numerator / max_sum_Rdeno
    F1 = 2 * P * R / (P + R)

    print('actor',actor, '  class',classs,'  action:',action,'  time',timee,'  location',location,'  object:',object,'  p:',P,'  r:',R,'  f1:',F1)
    return F1


if __name__ == '__main__':
    self_location = '../result/test_qwen.json'
    valid_location = 'C:/Users/sibin/worksp/COMPETITION/Causality Extraction/valid.json'

    start = time.time()

    self_data_dict = read_json(self_location)
    valid_data_dict = read_json(valid_location)
    num_cores = 15
    print(f'num_cores:{num_cores}')
    splited_self_data_dict_list = split_dict(self_data_dict,num_cores)

    # 创建进程池
    with mp.Pool(num_cores) as pool:
        # 并行处理每个子字典
        results = pool.starmap(re_result_dict, [(chunk, valid_data_dict) for chunk in splited_self_data_dict_list])

    pool.close()
    pool.join()

    re_F1(results)
    end_ = time.time()
    print('运行时间', convert_seconds(end_-start))



