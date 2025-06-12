import os
import signal
import sys

# from build.lib.pearl.SMTimer.check_time import logger

os.environ['ALL_PROXY'] = ''
os.environ['all_proxy'] = ''
import ast
import json
import random
import re
import time

from z3 import *
from z3.z3 import parse_smt2_string, Solver

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
from env_gai_6_llm_add_ce_predictor_docker_llm_embed import ConstraintSimplificationEnv_test

# from test_code_bert_4 import CodeEmbedder, CodeEmbedder_normalize
from bert_embedder_test import CodeEmbedder_normalize

from test_rl.predictor.smt_comp_NIA.bert_predictor_mask_llm import EnhancedClassifier
from test_rl.predictor.smt_comp_NIA.bert_predictor_2_mask_llm import EnhancedEightClassModelLargeInput
from test_rl.bert_predictor_2_mask import EnhancedEightClassModel
from test_rl.bert_predictor_mask import SimpleClassifier
from test_rl.test_script.utils import parse_smt2_in_parts, process_smt_lib_string, fetch_data_as_dict, \
    solve_and_measure_time, model_to_dict, load_dictionary, extract_variables_from_smt2_content, normalize_variables, \
    normalize_smt_str,MyException,timeout_handler
from test_rl.test_script.online_learning_break import online_learning
from loguru import logger

start = time.time()

def test_group():

    with open('/home/lz/PycharmProjects/Pearl/test_rl/test_solve/NIA/NIA.json', 'r') as file:
        solve_dict = json.load(file)

    info_name = 'info_dict_gai_6_normal_0503_pre_llm_llama3.1:70b_1200s_QF_NIA.txt'
    if not os.path.exists(info_name):
        # 文件不存在时，创建文件
        info_dict = {}
        with open(info_name, 'w') as file:
            json.dump(info_dict, file, indent=4)
        print(f'文件{info_name} 已创建。')
    else:
        info_dict = load_dictionary(info_name)
        print(f'文件已存在。')
    with open('/home/lz/PycharmProjects/Pearl/test_rl/predictor/smt_comp_NIA/QF_NIA_test.json', 'r') as file:

        result_dict = json.load(file)
    #随机打乱
    items = list(result_dict.items())
    random.shuffle(items)
    result_dict = dict(items)
    count = 0
    for key, value in result_dict.items():
        state = torch.tensor(np.load(value[0]))
        list1 = solve_dict[key]
        logger.info(f'{key} {list1}')
        if list1[0] == "sat" or list1[0] == "unknown":
            if list1[1] > 300 and key in result_dict.keys():
                count += 1
    print(count)


if __name__ == '__main__':
    test_group()
