import json
import re
import time
import random

from z3 import *

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
from env_docker import ConstraintSimplificationEnv_test

from code_bert_docker import CodeEmbedder
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
    if not os.path.exists('file_time_docker.txt'):
        # 文件不存在时，创建文件
        file_time = {}
        with open('file_time_docker.txt', 'w') as file:
            json.dump(file_time, file, indent=4)
        print(f"文件 file_time_docker.txt 已创建。")
    else:
        # 文件已存在
        print(f"文件已存在。")
    # file_time = {}
    # with open('file_time_docker.txt', 'w') as file:
    #     json.dump(file_time, file, indent=4)
    # info_dict = {}
    if not os.path.exists('info_dict_docker.txt'):
        # 文件不存在时，创建文件
        info_dict = {}
        with open('info_dict_docker.txt', 'w') as file:
            json.dump(info_dict, file, indent=4)
        print(f"文件 info_dict_docker.txt 已创建。")
    else:
        info_dict = load_dictionary('info_dict_docker.txt')
        print(f"文件已存在。")

    # file_time = {}
    # with open('file_time_docker.txt', 'w') as file:
    #     json.dump(file_time, file, indent=4)
    # info_dict = {}
    db_path = "/tmp/pycharm_project_20/test_rl/test_script/result_dictionary.db"
    table_name = 'result_dictionary'
    result_dict = fetch_data_as_dict(db_path, table_name)
    #打乱顺序
    items = list(result_dict.items())
    # 使用random.shuffle打乱列表
    random.shuffle(items)
    # 从打乱后的项列表创建一个新的字典
    result_dict = dict(items)
    for key, value in result_dict.items():
        list1 = json.loads(value)
        if list1[0] == "sat":
            if list1[1] > 100:
                print(key, value)
                if '/home/yy/Downloads/' in key:
                    file_path = key.replace('/home/yy/Downloads/', '/home/user/')
                elif '/home/nju/Downloads/' in key:
                    file_path = key.replace('/home/nju/Downloads/', '/home/user/')
                elif '/home/lz/baidudisk/' in key:
                    file_path = key.replace('/home/lz/baidudisk/', '/home/user/')
                else:
                    file_path = key
                file_time = load_dictionary('file_time_docker.txt')
                file_time[file_path] = 0
                with open('file_time_docker.txt', 'w') as file:
                    json.dump(file_time, file, indent=4)

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

                assertions = parse_smt2_string(smtlib_str)
                solver = Solver()
                for a in assertions:
                    solver.add(a)
                timeout = 999999999
                # timeout = 1000
                result, model, time_taken = solve_and_measure_time(solver, timeout)
                print(result,time_taken)
                result_list = [result, time_taken, timeout]
                # if result == sat:
                #     result = 'sat'
                # elif result == unknown:
                #     result = 'unknown'
                # else:
                #     result = 'unsat'

                # result_dict[filepath] = result_list
                if model:
                    result_list.append(model_to_dict(model))
                # file_path = key
                # file_path = '/home/lz/baidudisk/smt/gnu_angr.tar.gz/single_test/ginstall/ginstall307943'

                start_time = time.time()

                # variables = set()
                variables = extract_variables_from_smt2_content(smtlib_str)

                print("变量列表：")
                for v in variables:
                    print(v)
                embedder = CodeEmbedder()
                set_seed(0)
                # device = torch.device("cpu")

                env = ConstraintSimplificationEnv_test(embedder, assertions, len(variables), len(variables), smtlib_str,
                                                       file_path)
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
                # number_of_steps = 500
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
                    history_summarization_module=LSTMHistorySummarizationModule(
                        observation_dim=768,
                        action_dim=len(env.variables) + 1,
                        hidden_dim=768,
                        history_length=len(env.variables),  # 和完整结点数相同
                    ),
                    replay_buffer=FIFOOffPolicyReplayBuffer(10),
                    device_id=-1,
                )
                # 训练代理
                info = online_learning_with_break(
                    agent=agent,
                    env=env,
                    # number_of_steps=number_of_steps,
                    number_of_episodes=number_of_episodes,
                    # print_every_x_steps=100,
                    print_every_x_episodes=1,
                    record_period=record_period,
                    # learn_after_episode=True,
                )
                end_time = time.time()
                result_list.append(end_time - start_time)
                if env.step_count > 500:
                    result_list.append('failed')
                else:
                    result_list.append('succeed')
                    result_list.append(env.solve_time)
                    result_list.append(env.counterexamples_list[-1])
                info_dict[file_path] = result_list
                with open('info_dict_docker.txt', 'w') as file:
                    json.dump(info_dict, file, indent=4)
                # torch.save(info["return"], "BootstrappedDQN-LSTM-return.pt")
                # plt.plot(record_period * np.arange(len(info["return"])), info["return"], label="BootstrappedDQN-LSTM")
                # plt.legend()
                # plt.show()


if __name__ == '__main__':
    test_group()
