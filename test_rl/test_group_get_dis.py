import ast
import json
import random
import re
import time

import tqdm
from z3 import *

from pearl.SMTimer.KNN_Predictor import Predictor
from pearl.policy_learners.sequential_decision_making.soft_actor_critic import SoftActorCritic
# from pearl.replay_buffers.sequential_decision_making.bootstrap_replay_buffer import BootstrapReplayBuffer
from pearl.replay_buffers.sequential_decision_making.bootstrap_replay_buffer import FIFOOffPolicyReplayBuffer
from pearl.utils.functional_utils.experimentation.set_seed import set_seed
from pearl.action_representation_modules.identity_action_representation_module import IdentityActionRepresentationModule
from pearl.history_summarization_modules.lstm_history_summarization_module import LSTMHistorySummarizationModule
from pearl.utils.functional_utils.train_and_eval.online_learning import online_learning, online_learning_with_break
from pearl.pearl_agent import PearlAgent

import torch
import matplotlib.pyplot as plt
import numpy as np
from env import ConstraintSimplificationEnv_v3, ConstraintSimplificationEnv_test

from test_code_bert_4 import CodeEmbedder, CodeEmbedder_normalize
from test_rl.test_script.utils import parse_smt2_in_parts, process_smt_lib_string, fetch_data_as_dict, \
    solve_and_measure_time, model_to_dict, load_dictionary, extract_variables_from_smt2_content, normalize_variables

start = time.time()


# def extract_variables_from_smt2_content(content):
#     """
#     从 SMT2 格式的字符串内容中提取变量名。
#
#     参数:
#     - content: SMT2 格式的字符串内容。
#
#     返回:
#     - 变量名列表。
#     """
#     # 用于匹配 `(declare-fun ...)` 语句中的变量名的正则表达式
#     variable_pattern = re.compile(r'\(declare-fun\s+([^ ]+)')
#
#     # 存储提取的变量名
#     variables = []
#
#     # 按行分割字符串并迭代每一行
#     for line in content.splitlines():
#         # 在每一行中查找匹配的变量名
#         match = variable_pattern.search(line)
#         if match:
#             # 如果找到匹配项，则将变量名添加到列表中
#             variables.append(match.group(1).replace('|', ''))
#
#     return set(variables)


def visit(expr):
    if is_const(expr) and expr.decl().kind() == Z3_OP_UNINTERPRETED:
        # Add only uninterpreted functions (which represent variables)
        variables.add(str(expr))
    else:
        # Recursively visit children for composite expressions
        for child in expr.children():
            visit(child)


def test_group():
    db_path = '/home/lz/PycharmProjects/Pearl/test_rl/test_script/result_dictionary.db'
    table_name = 'result_dictionary'
    result_dict = fetch_data_as_dict(db_path, table_name)
    items = list(result_dict.items())
    random.shuffle(items)
    result_dict = dict(items)

    stats = {
        'sat': {'count': 0, 'percentage': 0},
        'unsat': {'count': 0, 'percentage': 0},
        'unknown': {'count': 0, 'percentage': 0},
        'times': {
            '1': {'count': 0, 'percentage': 0},
            '20': {'count': 0, 'percentage': 0},
            '50': {'count': 0, 'percentage': 0},
            '100': {'count': 0, 'percentage': 0},
            '200': {'count': 0, 'percentage': 0},
            '500': {'count': 0, 'percentage': 0}
        }
    }
    embedder = CodeEmbedder()
    # 遍历字典并统计数据
    total_entries = len(result_dict)
    features_list = []
    labels_list = []
    time_list = []
    for key, value in result_dict.items():
        if '/home/yy/Downloads/' in key:
            file_path = key.replace('/home/yy/Downloads/', '/home/lz/baidudisk/')
        elif '/home/nju/Downloads/' in key:
            file_path = key.replace('/home/nju/Downloads/', '/home/lz/baidudisk/')
        with open(file_path, 'r') as file:
            # 读取文件所有内容到一个字符串
            smtlib_str = file.read()
        # 解析字符串
        try:
            # 将JSON字符串转换为字典
            dict_obj = json.loads(smtlib_str)
            # print("转换后的字典：", dict_obj)
        except json.JSONDecodeError as e:
            print("解析错误：", e)
        #
        if 'smt-comp' in file_path:
            smtlib_str = dict_obj['smt_script']
        else:
            smtlib_str = dict_obj['script']
        embeddings = embedder.get_max_pooling_embedding(smtlib_str)
        features_list.append(embeddings)
        list1 = json.loads(value)
        category = list1[0]  # "sat", "unsat", 或 "unknown"
        time_value = list1[1]  # 消耗的时间

        if list1[0] == "sat":
            labels_list.append(0)
            if time_value <= 1:
                time_list.append(1)
                result_dict[key].append(1)
            elif time_value <= 20:
                time_list.append(2)
                result_dict[key].append(2)
            elif time_value <= 50:
                time_list.append(3)
                result_dict[key].append(3)
            elif time_value <= 100:
                time_list.append(4)
                result_dict[key].append(4)
            elif time_value <= 200:
                time_list.append(5)
                result_dict[key].append(5)
            elif time_value <= 500:
                time_list.append(6)
                result_dict[key].append(6)
            else:
                time_list.append(7)
                result_dict[key].append(7)
        else:
            labels_list.append(1)
            # 没有时间
            time_list.append(0)
            result_dict[key].append(0)

        # 更新类别统计
        stats[category]['count'] += 1

        # 更新时间统计
        if time_value <= 1:
            time_key = '1'
        elif time_value <= 20:
            time_key = '20'
        elif time_value <= 50:
            time_key = '50'
        elif time_value <= 100:
            time_key = '100'
        elif time_value <= 200:
            time_key = '200'
        elif time_value <= 500:
            time_key = '500'
        else:
            time_key = '500'  # 超过500的时间归类到500
        stats['times'][time_key]['count'] += 1
    # 计算百分比
    for category in stats:
        if category != 'times':
            stats[category]['percentage'] = (stats[category]['count'] / total_entries) * 100

    for time_key in stats['times']:
        stats['times'][time_key]['percentage'] = (stats['times'][time_key]['count'] / total_entries) * 100

    # 打印统计结果
    print("Category Statistics:")
    for category, data in stats.items():
        if category != 'times':
            print(f"  {category}: Count = {data['count']}, Percentage = {data['percentage']:.2f}%")

    print("\nTime Statistics:")
    for time_key, data in stats['times'].items():
        print(f"  Time <= {time_key}: Count = {data['count']}, Percentage = {data['percentage']:.2f}%")

    features_array = np.array([t.numpy() if isinstance(t, torch.Tensor) else t for t in features_list])
    labels_array = np.array(labels_list)
    time_array = np.array(time_list)
    # 保存特征和标签数组到文件
    np.save('features.npy', features_array)
    np.save('labels.npy', labels_array)
    np.save('time.npy', time_array)
    # with open('result_dict_time.txt', 'w') as file:
    #     # 使用json.dump()将字典保存到文件
    #     json.dump(result_dict, file, indent=4)


def test_group_2():
    predictor = Predictor('KNN')

    db_path = '/home/lz/PycharmProjects/Pearl/test_rl/test_script/result_dictionary.db'
    table_name = 'result_dictionary'
    result_dict = fetch_data_as_dict(db_path, table_name)
    items = list(result_dict.items())
    random.shuffle(items)
    result_dict = dict(items)

    stats = {
        'sat': {'count': 0, 'percentage': 0},
        'unsat': {'count': 0, 'percentage': 0},
        'unknown': {'count': 0, 'percentage': 0},
        'times': {
            '1': {'count': 0, 'percentage': 0},
            '20': {'count': 0, 'percentage': 0},
            '50': {'count': 0, 'percentage': 0},
            '100': {'count': 0, 'percentage': 0},
            '200': {'count': 0, 'percentage': 0},
            '500': {'count': 0, 'percentage': 0}
        }
    }
    embedder = CodeEmbedder()
    # 遍历字典并统计数据
    total_entries = len(result_dict)
    features_list = []
    labels_list = []
    time_list = []
    count = 0
    for key, value in result_dict.items():
        if '/home/yy/Downloads/' in key:
            file_path = key.replace('/home/yy/Downloads/', '/home/lz/baidudisk/')
        elif '/home/nju/Downloads/' in key:
            file_path = key.replace('/home/nju/Downloads/', '/home/lz/baidudisk/')
        with open(file_path, 'r') as file:
            # 读取文件所有内容到一个字符串
            smtlib_str = file.read()
        # 解析字符串
        try:
            # 将JSON字符串转换为字典
            dict_obj = json.loads(smtlib_str)
            # print("转换后的字典：", dict_obj)
        except json.JSONDecodeError as e:
            print("解析错误：", e)
        #
        if 'smt-comp' in file_path:
            smtlib_str = dict_obj['smt_script']
        else:
            smtlib_str = dict_obj['script']
        p = predictor.predict(smtlib_str)
        list1 = json.loads(value)
        category = list1[0]  # "sat", "unsat", 或 "unknown"
        time_value = list1[1]  # 消耗的时间

        if list1[0] == "sat" and p == 0:
            count += 1
        elif list1[0] == "unsat" and p == 1:
            count += 1
        elif list1[0] == "unknown" and p == 1:
            count += 1
    print(count)
    print(len(result_dict))
    print(count / len(result_dict))


def test_group_3():
    db_path = '/home/lz/PycharmProjects/Pearl/test_rl/test_script/result_dictionary.db'
    table_name = 'result_dictionary'
    result_dict = fetch_data_as_dict(db_path, table_name)
    items = list(result_dict.items())
    random.shuffle(items)
    result_dict = dict(items)

    stats = {
        'sat': {'count': 0, 'percentage': 0},
        'unsat': {'count': 0, 'percentage': 0},
        'unknown': {'count': 0, 'percentage': 0},
        'times': {
            '1': {'count': 0, 'percentage': 0},
            '20': {'count': 0, 'percentage': 0},
            '50': {'count': 0, 'percentage': 0},
            '100': {'count': 0, 'percentage': 0},
            '200': {'count': 0, 'percentage': 0},
            '500': {'count': 0, 'percentage': 0}
        }
    }
    embedder = CodeEmbedder()
    # 遍历字典并统计数据
    total_entries = len(result_dict)
    features_list = []
    labels_list = []
    time_list = []
    for key, value in result_dict.items():
        if '/home/yy/Downloads/' in key:
            file_path = key.replace('/home/yy/Downloads/', '/home/lz/baidudisk/')
        elif '/home/nju/Downloads/' in key:
            file_path = key.replace('/home/nju/Downloads/', '/home/lz/baidudisk/')
        with open(file_path, 'r') as file:
            # 读取文件所有内容到一个字符串
            smtlib_str = file.read()
        # 解析字符串
        try:
            # 将JSON字符串转换为字典
            dict_obj = json.loads(smtlib_str)
            # print("转换后的字典：", dict_obj)
        except json.JSONDecodeError as e:
            print("解析错误：", e)
        #
        if 'smt-comp' in file_path:
            smtlib_str = dict_obj['smt_script']
        else:
            smtlib_str = dict_obj['script']
        list1 = json.loads(value)
        category = list1[0]  # "sat", "unsat", 或 "unknown"
        time_value = list1[1]  # 消耗的时间

        value = ast.literal_eval(value)
        result_dict[key] = value
        if list1[0] == "sat":
            labels_list.append(0)
            if time_value <= 1:
                time_list.append(1)
                result_dict[key].append(1)
            elif time_value <= 20:
                time_list.append(2)
                result_dict[key].append(2)
            elif time_value <= 50:
                time_list.append(3)
                result_dict[key].append(3)
            elif time_value <= 100:
                time_list.append(4)
                result_dict[key].append(4)
            elif time_value <= 200:
                time_list.append(5)
                result_dict[key].append(5)
            elif time_value <= 500:
                time_list.append(6)
                result_dict[key].append(6)
            else:
                time_list.append(7)
                result_dict[key].append(7)
        else:
            labels_list.append(1)
            # 没有时间
            time_list.append(0)
            result_dict[key].append(0)

        # 更新类别统计
        stats[category]['count'] += 1

        # 更新时间统计
        if time_value <= 1:
            time_key = '1'
        elif time_value <= 20:
            time_key = '20'
        elif time_value <= 50:
            time_key = '50'
        elif time_value <= 100:
            time_key = '100'
        elif time_value <= 200:
            time_key = '200'
        elif time_value <= 500:
            time_key = '500'
        else:
            time_key = '500'  # 超过500的时间归类到500
        stats['times'][time_key]['count'] += 1
    # 计算百分比
    for category in stats:
        if category != 'times':
            stats[category]['percentage'] = (stats[category]['count'] / total_entries) * 100

    for time_key in stats['times']:
        stats['times'][time_key]['percentage'] = (stats['times'][time_key]['count'] / total_entries) * 100

    # 打印统计结果
    print("Category Statistics:")
    for category, data in stats.items():
        if category != 'times':
            print(f"  {category}: Count = {data['count']}, Percentage = {data['percentage']:.2f}%")

    print("\nTime Statistics:")
    for time_key, data in stats['times'].items():
        print(f"  Time <= {time_key}: Count = {data['count']}, Percentage = {data['percentage']:.2f}%")

    with open('result_dict_time.txt', 'w') as file:
        # 使用json.dump()将字典保存到文件
        json.dump(result_dict, file, indent=4)


def test_group_4():
    with open('/home/lz/sibyl_3/src/networks/info_dict_predictor.txt', 'r') as file:
        result_dict = json.load(file)
    # items = list(result_dict.items())
    # random.shuffle(items)
    # result_dict = dict(items)

    stats = {
        'sat': {'count': 0, 'percentage': 0},
        'unsat': {'count': 0, 'percentage': 0},
        'unknown': {'count': 0, 'percentage': 0},
        'times': {
            '1': {'count': 0, 'percentage': 0},
            '20': {'count': 0, 'percentage': 0},
            '50': {'count': 0, 'percentage': 0},
            '100': {'count': 0, 'percentage': 0},
            '200': {'count': 0, 'percentage': 0},
            '500': {'count': 0, 'percentage': 0}
        }
    }
    # 遍历字典并统计数据
    total_entries = len(result_dict)
    features_list = []
    labels_list = []
    time_list = []
    for key, value in result_dict.items():
        if '/home/yy/Downloads/' in key:
            file_path = key.replace('/home/yy/Downloads/', '/home/lz/baidudisk/')
        elif '/home/nju/Downloads/' in key:
            file_path = key.replace('/home/nju/Downloads/', '/home/lz/baidudisk/')
        else:
            file_path = key
        with open(file_path, 'r') as file:
            # 读取文件所有内容到一个字符串
            smtlib_str = file.read()
        # 解析字符串
        try:
            # 将JSON字符串转换为字典
            dict_obj = json.loads(smtlib_str)
            # print("转换后的字典：", dict_obj)
        except json.JSONDecodeError as e:
            print("解析错误：", e)
        #
        if 'smt-comp' in file_path:
            smtlib_str = dict_obj['smt_script']
        else:
            smtlib_str = dict_obj['script']
        # list1 = json.loads(value)
        list1 = value
        category = list1[0]  # "sat", "unsat", 或 "unknown"
        time_value = list1[1]  # 消耗的时间

        # value = ast.literal_eval(value)
        result_dict[key] = value
        if list1[0] == "sat":
            labels_list.append(0)
            if time_value <= 1:
                time_list.append(1)
                result_dict[key].append(1)
            elif time_value <= 20:
                time_list.append(2)
                result_dict[key].append(2)
            elif time_value <= 50:
                time_list.append(3)
                result_dict[key].append(3)
            elif time_value <= 100:
                time_list.append(4)
                result_dict[key].append(4)
            elif time_value <= 200:
                time_list.append(5)
                result_dict[key].append(5)
            elif time_value <= 500:
                time_list.append(6)
                result_dict[key].append(6)
            else:
                time_list.append(7)
                result_dict[key].append(7)
        else:
            labels_list.append(1)
            # 没有时间
            time_list.append(0)
            result_dict[key].append(0)

        # 更新类别统计
        stats[category]['count'] += 1

        # 更新时间统计
        if time_value <= 1:
            time_key = '1'
        elif time_value <= 20:
            time_key = '20'
        elif time_value <= 50:
            time_key = '50'
        elif time_value <= 100:
            time_key = '100'
        elif time_value <= 200:
            time_key = '200'
        elif time_value <= 500:
            time_key = '500'
        else:
            time_key = '500'  # 超过500的时间归类到500
        stats['times'][time_key]['count'] += 1
    # 计算百分比
    for category in stats:
        if category != 'times':
            stats[category]['percentage'] = (stats[category]['count'] / total_entries) * 100

    for time_key in stats['times']:
        stats['times'][time_key]['percentage'] = (stats['times'][time_key]['count'] / total_entries) * 100

    # 打印统计结果
    print("Category Statistics:")
    for category, data in stats.items():
        if category != 'times':
            print(f"  {category}: Count = {data['count']}, Percentage = {data['percentage']:.2f}%")

    print("\nTime Statistics:")
    for time_key, data in stats['times'].items():
        print(f"  Time <= {time_key}: Count = {data['count']}, Percentage = {data['percentage']:.2f}%")

    with open('/home/lz/sibyl_3/src/networks/result_dict_time.txt', 'w') as file:
        # 使用json.dump()将字典保存到文件
        json.dump(result_dict, file, indent=4)


def test_group_bert():
    with open('result_dict_time_pre.txt', 'r') as file:
        result_dict = json.load(file)

    stats = {
        'sat': {'count': 0, 'percentage': 0},
        'unsat': {'count': 0, 'percentage': 0},
        'unknown': {'count': 0, 'percentage': 0},
        'times': {
            '1': {'count': 0, 'percentage': 0},
            '20': {'count': 0, 'percentage': 0},
            '50': {'count': 0, 'percentage': 0},
            '100': {'count': 0, 'percentage': 0},
            '200': {'count': 0, 'percentage': 0},
            '500': {'count': 0, 'percentage': 0}
        }
    }
    embedder = CodeEmbedder()
    # 遍历字典并统计数据
    total_entries = len(result_dict)
    features_list = []
    labels_list = []
    time_list = []
    for (i, (key, value)) in enumerate(tqdm.tqdm(result_dict.items())):
        file_path = key
        with open(file_path, 'r') as file:
            # 读取文件所有内容到一个字符串
            smtlib_str = file.read()
        # 解析字符串
        try:
            # 将JSON字符串转换为字典
            dict_obj = json.loads(smtlib_str)
            # print("转换后的字典：", dict_obj)
        except json.JSONDecodeError as e:
            print("解析错误：", e)
        #
        if 'smt-comp' in file_path:
            smtlib_str = dict_obj['smt_script']
        else:
            smtlib_str = dict_obj['script']
        embeddings = embedder.get_max_pooling_embedding(smtlib_str)
        features_list.append(embeddings)
        list1 = value
        category = list1[0]  # "sat", "unsat", 或 "unknown"
        time_value = list1[1]  # 消耗的时间
        result_dict[key].append(i)
        if list1[0] == "sat":
            labels_list.append(0)
            if time_value <= 1:
                time_list.append(1)
                result_dict[key].append(1)
            elif time_value <= 20:
                time_list.append(2)
                result_dict[key].append(2)
            elif time_value <= 50:
                time_list.append(3)
                result_dict[key].append(3)
            elif time_value <= 100:
                time_list.append(4)
                result_dict[key].append(4)
            elif time_value <= 200:
                time_list.append(5)
                result_dict[key].append(5)
            elif time_value <= 500:
                time_list.append(6)
                result_dict[key].append(6)
            else:
                time_list.append(7)
                result_dict[key].append(7)
        else:
            labels_list.append(1)
            # 没有时间
            time_list.append(0)
            result_dict[key].append(0)

        # 更新类别统计
        stats[category]['count'] += 1

        # 更新时间统计
        if time_value <= 1:
            time_key = '1'
        elif time_value <= 20:
            time_key = '20'
        elif time_value <= 50:
            time_key = '50'
        elif time_value <= 100:
            time_key = '100'
        elif time_value <= 200:
            time_key = '200'
        elif time_value <= 500:
            time_key = '500'
        else:
            time_key = '500'  # 超过500的时间归类到500
        stats['times'][time_key]['count'] += 1
    # 计算百分比
    for category in stats:
        if category != 'times':
            stats[category]['percentage'] = (stats[category]['count'] / total_entries) * 100

    for time_key in stats['times']:
        stats['times'][time_key]['percentage'] = (stats['times'][time_key]['count'] / total_entries) * 100

    # 打印统计结果
    print("Category Statistics:")
    for category, data in stats.items():
        if category != 'times':
            print(f"  {category}: Count = {data['count']}, Percentage = {data['percentage']:.2f}%")

    print("\nTime Statistics:")
    for time_key, data in stats['times'].items():
        print(f"  Time <= {time_key}: Count = {data['count']}, Percentage = {data['percentage']:.2f}%")

    features_array = np.array([t.numpy() if isinstance(t, torch.Tensor) else t for t in features_list])
    labels_array = np.array(labels_list)
    time_array = np.array(time_list)
    # 保存特征和标签数组到文件
    np.save('features.npy', features_array)
    np.save('labels.npy', labels_array)
    np.save('time.npy', time_array)
    with open('result_dict_time_pre_order.txt', 'w') as file:
        # 使用json.dump()将字典保存到文件
        json.dump(result_dict, file, indent=4)


def test_group_bert_normalize():
    with open('result_dict_time_pre_order.txt', 'r') as file:
        result_dict = json.load(file)

    embedder = CodeEmbedder_normalize()
    # 遍历字典并统计数据
    features_list = []
    for (i, (key, value)) in enumerate(tqdm.tqdm(result_dict.items())):
        file_path = key
        # print(key)
        with open(file_path, 'r') as file:
            # 读取文件所有内容到一个字符串
            smtlib_str = file.read()
        # 解析字符串
        try:
            # 将JSON字符串转换为字典
            dict_obj = json.loads(smtlib_str)
            # print("转换后的字典：", dict_obj)
        except json.JSONDecodeError as e:
            print("解析错误：", e)
        #
        if 'smt-comp' in file_path:
            smtlib_str = dict_obj['smt_script']
        else:
            smtlib_str = dict_obj['script']
        variables = extract_variables_from_smt2_content(smtlib_str)
        smtlib_str = normalize_variables(smtlib_str, variables)
        embeddings = embedder.get_max_pooling_embedding(smtlib_str)
        features_list.append(embeddings)

    features_array = np.array([t.numpy() if isinstance(t, torch.Tensor) else t for t in features_list])
    # 保存特征和标签数组到文件
    np.save('features_normal.npy', features_array)

    # with open('result_dict_time_pre_order.txt', 'w') as file:
    #     # 使用json.dump()将字典保存到文件
    #     json.dump(result_dict, file, indent=4)
def test_group_bert_normalize_1by1():
    with open('result_dict_time_pre_order.txt', 'r') as file:
        result_dict = json.load(file)

    embedder = CodeEmbedder_normalize()

    # 遍历字典并统计数据
    for (i, (file_path, _)) in enumerate(tqdm.tqdm(result_dict.items())):
        # 读取文件内容
        with open(file_path, 'r') as file:
            smtlib_str = file.read()

        # 解析字符串为JSON
        try:
            dict_obj = json.loads(smtlib_str)
            if 'smt-comp' in file_path:
                smtlib_str = dict_obj['smt_script']
            else:
                smtlib_str = dict_obj['script']
        except json.JSONDecodeError as e:
            print("解析错误：", e)
            continue  # 如果解析出错，跳过当前文件，继续处理下一个文件

        # 处理smtlib_str
        variables = extract_variables_from_smt2_content(smtlib_str)
        smtlib_str = normalize_variables(smtlib_str, variables)
        embeddings = embedder.get_max_pooling_embedding(smtlib_str)

        # 将embeddings转换为numpy数组，如果它是一个torch.Tensor
        embeddings_array = embeddings.detach().numpy() if isinstance(embeddings, torch.Tensor) else np.array(embeddings)

        # 保存当前文件的特征到单独的Numpy文件中
        np.save(f'features/features_normal_{i}.npy', embeddings_array)
        del embeddings_array
        # 如果需要的话，可以在这里清空 embeddings_array 以节省内存


if __name__ == '__main__':
    test_group_bert_normalize_1by1()
    # loaded_features = np.load('features.npy')
    # loaded_labels = np.load('labels.npy')
    # print(loaded_features.shape)
