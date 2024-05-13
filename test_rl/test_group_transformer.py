import json
import random
import re
import time

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

from test_code_bert_4 import CodeEmbedder
from test_rl.test_script.utils import parse_smt2_in_parts, process_smt_lib_string, fetch_data_as_dict, \
    solve_and_measure_time, model_to_dict, load_dictionary

start = time.time()


def extract_variables_from_smt2_content(content):
    """
    从 SMT2 格式的字符串内容中提取变量名。

    参数:
    - content: SMT2 格式的字符串内容。

    返回:
    - 变量名列表。
    """
    # 用于匹配 `(declare-fun ...)` 语句中的变量名的正则表达式
    variable_pattern = re.compile(r'\(declare-fun\s+([^ ]+)')

    # 存储提取的变量名
    variables = []

    # 按行分割字符串并迭代每一行
    for line in content.splitlines():
        # 在每一行中查找匹配的变量名
        match = variable_pattern.search(line)
        if match:
            # 如果找到匹配项，则将变量名添加到列表中
            variables.append(match.group(1).replace('|', ''))

    return set(variables)


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
    with open('result_dict.txt', 'w') as file:
        json.dump(result_dict, file, indent=4)
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


if __name__ == '__main__':
    test_group()
    # loaded_features = np.load('features.npy')
    # loaded_labels = np.load('labels.npy')
    # print(loaded_features.shape)