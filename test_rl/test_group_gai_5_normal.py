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

start = time.time()


def test_group():
    info_name = 'info_dict_gai_4_normal911_2.txt'
    if not os.path.exists(info_name):
        # 文件不存在时，创建文件
        info_dict = {}
        with open(info_name, 'w') as file:
            json.dump(info_dict, file, indent=4)
        print(f'文件{info_name} 已创建。')
    with open(info_name, 'r') as file:
        info_dict = json.load(file)
    with open('/home/lz/PycharmProjects/Pearl/test_rl/test_script/result_dict_no_increment.txt', 'r') as file:
        result_dict = json.load(file)
    #添加一个限制
    # with open('info_bit_dict.txt', 'r') as file:
    #     info_bit_dict = json.load(file)
    # check_list = []
    # for k,v in info_bit_dict.items():
    #     if int(k) <= 32:
    #         for v1 in v:
    #             check_list.append(v1)
    items = list(result_dict.items())
    random.shuffle(items)
    result_dict = dict(items)
    for key, value in result_dict.items():
        list1 = value
        if list1[0] == "sat":
            if list1[1] > 20:
                # if '/who/who86404' in key:
                print(key, value)
                file_path = key
                if file_path not in info_dict.keys():
                    #跳过无法处理的文件
                    if 'gnu_angr.tar.gz/single_test/cat/cat43772' in file_path:
                        continue
                    with open(file_path, 'r') as file:
                        # 读取文件所有内容到一个字符串
                        smtlib_str = file.read()
                    print(type(smtlib_str))
                    # # 解析字符串
                    # try:
                    #     # 将JSON字符串转换为字典
                    #     dict_obj = json.loads(smtlib_str)
                    #     # print("转换后的字典：", dict_obj)
                    # except json.JSONDecodeError as e:
                    #     print("解析错误：", e)
                    # #
                    # if 'smt-comp' in file_path:
                    #     smtlib_str = dict_obj['smt_script']
                    # else:
                    #     smtlib_str = dict_obj['script']
                    # variables = set()
                    # variables = extract_variables_from_smt2_content(smtlib_str)
                    # smtlib_str = normalize_variables(smtlib_str, variables)
                    smtlib_str, var_dict, constant_list = normalize_smt_str(smtlib_str)
                    if len(var_dict) > 20:
                        continue
                    assertions = parse_smt2_string(smtlib_str)
                    solver = Solver()
                    for a in assertions:
                        solver.add(a)
                    timeout = 999999999
                    # timeout = 1000

                    #先取消求解
                    result, model, time_taken = solve_and_measure_time(solver, timeout)
                    print(result, time_taken)
                    result_list = [result, time_taken, timeout]

                    if model:
                        result_list.append(model_to_dict(model))
                    print(result_list[-1])

                    start_time = time.time()

                    embedder = CodeEmbedder_normalize()
                    set_seed(0)
                    # device = torch.device("cpu")
                    # 更改了预测器
                    model = SimpleClassifier()
                    model_path = 'bert_predictor_mask_best.pth'  # 或者 'bert_predictor_mask_final.pth'
                    state_dict = torch.load(model_path)
                    model.load_state_dict(state_dict)
                    model.eval()
                    model_time = EnhancedEightClassModel()
                    model_time.load_state_dict(torch.load('bert_predictor_2_mask_best_model.pth'))
                    model_time.eval()
                    env = ConstraintSimplificationEnv_test(embedder, assertions, model, model_time, smtlib_str,
                                                           file_path, var_dict, constant_list)
                    observation, action_space = env.reset()
                    action_representation_module = IdentityActionRepresentationModule(
                        max_number_actions=action_space.n,
                        representation_dim=action_space.action_dim,
                    )
                    # action_representation_module = OneHotActionTensorRepresentationModule(
                    #     max_number_actions=len(env.variables)*20000,
                    # )
                    # action_representation_module = IdentityActionRepresentationModule(
                    #     max_number_actions=len(variables)*20000,
                    #     representation_dim=action_space.action_dim,
                    # )
                    # experiment code
                    number_of_steps = 500
                    number_of_episodes = 1
                    record_period = 1
                    # 创建强化学习代理
                    print(len(env.variables))
                    agent = PearlAgent(
                        policy_learner=SoftActorCritic(
                            state_dim=768,
                            action_space=action_space,
                            actor_hidden_dims=[768, 512, 128],
                            critic_hidden_dims=[768, 512, 128],
                            action_representation_module=action_representation_module,
                        ),
                        # history_summarization_module=StackingHistorySummarizationModule(
                        #     observation_dim=768,
                        #     action_dim=len(env.variables) + 1,
                        #     history_length=len(env.variables),
                        # ),
                        history_summarization_module=LSTMHistorySummarizationModule(
                            observation_dim=768,
                            action_dim=4,
                            hidden_dim=768,
                            history_length=len(env.variables),  # 和完整结点数相同
                        ),
                        replay_buffer=FIFOOffPolicyReplayBuffer(10),
                        device_id=-1,
                    )
                    # 训练代理
                    info = online_learning(
                        agent=agent,
                        env=env,
                        number_of_episodes=number_of_episodes,
                        print_every_x_episodes=1,
                        record_period=record_period,
                        # learn_after_episode=True,
                    )
                    end_time = time.time()
                    result_list.append(end_time - start_time)

                    if env.solve_time == 0:
                        result_list.append('failed')
                    else:
                        result_list.append('succeed')
                        result_list.append(env.solve_time)
                        result_list.append(env.counterexamples_list[-1])
                    result_list.append(env.counterexamples_list)
                    info_dict[file_path] = result_list
                    with open('info_dict_gai_4_normal729.txt', 'w') as file:
                        json.dump(info_dict, file, indent=4)
                    del agent
                    del env
                    torch.cuda.empty_cache()
                    # torch.save(info["return"], "BootstrappedDQN-LSTM-return.pt")
                    # plt.plot(record_period * np.arange(len(info["return"])), info["return"], label="BootstrappedDQN-LSTM")
                    # plt.legend()
                    # plt.show()


if __name__ == '__main__':
    test_group()
