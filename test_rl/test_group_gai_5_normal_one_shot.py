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
    if not os.path.exists('info_dict_gai_4_normal910.txt'):
        # 文件不存在时，创建文件
        info_dict = {}
        with open('info_dict_gai_4_normal910.txt', 'w') as file:
            json.dump(info_dict, file, indent=4)
        print(f"文件 info_dict_gai_4_normal910.txt 已创建。")
    with open('info_dict_gai_4_normal910.txt', 'r') as file:
        info_dict = json.load(file)

    solver = Solver()  # 定义变量
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
    const_6 = BitVecVal(6, 32)
    const_2_bit = BitVecVal(2, 32)
    const_FF = BitVecVal(0xFF, 32)
    const_3 = BitVecVal(3, 32)
    const_73 = BitVecVal(73, 32)
    const_1F = BitVecVal(0x1F, 32)
    const_3_bit = BitVecVal(3, 32)
    const_71 = BitVecVal(71, 32)
    const_12 = BitVecVal(12, 32)
    const_128 = BitVecVal(128, 32)
    const_7 = BitVecVal(7, 32)
    const_8 = BitVecVal(8, 32)

    solver = Solver()

    # 复杂的模运算、位运算和条件分支，难以求解
    solver.add((x ^ (y * z) + (z * const_57 % (x + const_2))) * (y & (x | z)) % const_1024 == (x * y) % const_511)

    # 额外的条件分支，当 x, y, z 取某些值时，这些分支会变得简单
    solver.add(
        If((x + y) % const_256 == const_128, z * (x + y) % const_97 == const_15, z * (x - y) % const_97 == const_42))

    # 增加一些位运算和非线性组合
    solver.add(((x * y * z) % (x + 1)) * ((x ^ y) + (z >> 1)) > const_5000)

    # 添加依赖于某些具体值的条件，使问题在赋值后简化
    solver.add(If(x == const_123456 and y == const_789012 and z == const_345678,
                  x + y + z == const_1253456,
                  x + y + z > const_1000000))

    # 添加一个复杂的等式约束，使问题在未赋值时难解
    solver.add(((x * y * z) % (x * y + 1)) * ((z ^ (x + y)) % const_97) == 0)

    # 复杂表达式
    complex_expr = ((((x * const_6) ^ (x >> const_2_bit)) & const_FF) * ((x | const_3) % const_73) + (
                (x & const_1F) << const_3_bit)) % const_71 / 2

    solver.add(
        ((x * x * y * z) % ((x * y * const_12) % (x % y + z))) * ((x * z) % ((x * const_128) % const_7)) + y * y ==
        (complex_expr * complex_expr) * z * ((x * z) % ((x * x) % (x - const_7 * y + z))) *
        ((z * ((x * const_12) % const_7)) % const_8) * ((x * const_3) % ((x * const_128) % const_7)))

    smtlib_str = solver.to_smt2()
    print(type(smtlib_str))
    try:
        smtlib_str, var_dict, constant_list = normalize_smt_str(smtlib_str)
    except:
        print('error')
        var_dict, constant_list = None, None
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
                                           'test1111111111', var_dict, constant_list)
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
    # info_dict = {}
    # file_path = 'test1111111111'
    # info_dict[file_path] = result_list
    with open('info_dict_gai_4_normal910.txt', 'w') as file:
        info_dict[len(info_dict)] = result_list
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
