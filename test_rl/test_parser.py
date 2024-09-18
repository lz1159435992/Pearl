import ast
import json
import random
import re
import time

from z3 import *

from pearl.policy_learners.sequential_decision_making.soft_actor_critic import SoftActorCritic
# from pearl.replay_buffers.sequential_decision_making.bootstrap_replay_buffer import BootstrapReplayBuffer
from pearl.replay_buffers.sequential_decision_making.bootstrap_replay_buffer import FIFOOffPolicyReplayBuffer
from pearl.utils.functional_utils.experimentation.set_seed import set_seed
from pearl.action_representation_modules.identity_action_representation_module import IdentityActionRepresentationModule
from pearl.history_summarization_modules.lstm_history_summarization_module import LSTMHistorySummarizationModule
from pearl.history_summarization_modules.stacking_history_summarization_module import StackingHistorySummarizationModule
# from pearl.utils.functional_utils.train_and_eval.online_learning import online_learning, online_learning_with_break
from pearl.pearl_agent import PearlAgent

import torch
import matplotlib.pyplot as plt
import numpy as np
from env_gai_4 import ConstraintSimplificationEnv_test

# from test_code_bert_4 import CodeEmbedder, CodeEmbedder_normalize
from bert_embedder_test import CodeEmbedder_normalize
from test_rl.bert_predictor_2_mask import EnhancedEightClassModel
from test_rl.bert_predictor_mask import SimpleClassifier
from test_rl.test_script.utils import parse_smt2_in_parts, process_smt_lib_string, fetch_data_as_dict, \
    solve_and_measure_time, model_to_dict, load_dictionary, extract_variables_from_smt2_content, normalize_variables, \
    normalize_smt_str
from test_rl.test_script.online_learning_break import online_learning
info_name = 'info_dict_gai_4_normal99.txt'
if not os.path.exists(info_name):
    # 文件不存在时，创建文件
    info_dict = {}
    with open(info_name, 'w') as file:
        json.dump(info_dict, file, indent=4)
    print(f'文件{info_name} 已创建。')
with open(info_name, 'r') as file:
    info_dict = json.load(file)
with open('/home/lz/PycharmProjects/Pearl/test_rl/ge_cons/auto_gen.txt', 'r') as file:
    result_dict = json.load(file)
for k, v in result_dict.items():
    smtlib_str = v
    print(type(smtlib_str))

    # 定义位向量
    x = BitVec('x', 32)
    y = BitVec('y', 32)
    z = BitVec('z', 32)

    # 定义常量为位向量
    const_57 = BitVecVal(57, 32)
    const_2 = BitVecVal(2, 32)
    const_1024 = BitVecVal(1024, 32)
    const_511 = BitVecVal(511, 32)
    const_256 = BitVecVal(256, 32)
    const_128 = BitVecVal(128, 32)
    const_97 = BitVecVal(97, 32)
    const_15 = BitVecVal(15, 32)
    const_42 = BitVecVal(42, 32)
    const_123456 = BitVecVal(123456, 32)
    const_789012 = BitVecVal(789012, 32)
    const_345678 = BitVecVal(345678, 32)
    const_1253456 = BitVecVal(1253456, 32)
    const_1000000 = BitVecVal(1000000, 32)
    const_5000 = BitVecVal(5000, 32)

    solver = Solver()

    # 复杂的模运算、位运算和条件分支，难以求解
    solver.add((x ^ (y * z) + (z * const_57 % (x + const_2))) * (y & (x | z)) % const_1024 == (x * y) % const_511)

    # 额外的条件分支，当 x, y, z 取某些值时，这些分支会变得简单
    solver.add(
        If((x + y) % const_256 == const_128, z * (x + y) % const_97 == const_15, z * (x - y) % const_97 == const_42))

    # 增加一些位运算和非线性组合
    solver.add(((x * y * z) % (x + const_2)) * ((x ^ y) + (z >> 1)) > const_5000)

    # 添加依赖于某些具体值的条件，使问题在赋值后简化
    solver.add(If(x == const_123456 and y == const_789012 and z == const_345678,
                  x + y + z == const_1253456,
                  x + y + z > const_1000000))

    # 添加一个复杂的等式约束，使问题在未赋值时难解
    solver.add(((x * y * z) % (x * y + const_2)) * ((z ^ (x + y)) % const_97) == 0)

    smtlib_str = solver.to_smt2()

    smtlib_str, var_dict, constant_list = normalize_smt_str(smtlib_str)