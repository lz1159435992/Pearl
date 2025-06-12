import re


def z3_smtlib_to_pysmt_compatible(smtlib_str: str, logic: str = "QF_NIA") -> str:
    """
    将 Z3 生成的 SMT-LIB2 字符串转换为 PySMT 可接受的格式。

    参数:
        smtlib_str: Z3 的 to_smt2() 生成的字符串。
        logic: SMT-LIB 逻辑（默认 QF_NIA，整数算术）。

    返回:
        PySMT 可解析的 SMT-LIB 字符串。
    """

    # 设置标准逻辑前缀（如果 Z3 没有提供）
    if not smtlib_str.strip().startswith("(set-logic"):
        smtlib_str = f"(set-logic {logic})\n" + smtlib_str

    # 替换 mod/div 成 PySMT 兼容形式（注意：PySMT 支持 mod/div，但用标准名）
    smtlib_str = re.sub(r'\(mod ', '(mod ', smtlib_str)
    smtlib_str = re.sub(r'\(div ', '(div ', smtlib_str)

    # 删除 Z3 特有语法，例如 (_ some-op ...)
    smtlib_str = re.sub(r'\(_ ([a-zA-Z0-9_\-]+) ', r'(\1 ', smtlib_str)

    # 删除 set-info / set-option 等非必需项
    smtlib_str = re.sub(r'\(set-info [^\)]+\)', '', smtlib_str)
    smtlib_str = re.sub(r'\(set-option [^\)]+\)', '', smtlib_str)

    # 删除空行和多余空格
    smtlib_str = "\n".join(line.strip() for line in smtlib_str.strip().splitlines() if line.strip())

    return smtlib_str


import os
import signal
import sys

from pysmt.shortcuts import write_smtlib

from test_rl.test_script.time import smtlib_str, file_path

os.environ['ALL_PROXY'] = ''
os.environ['all_proxy'] = ''
import ast
import json
import random
import re
import time
from loguru import logger
from z3 import *
from z3.z3 import parse_smt2_string, Solver, Int, And

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
from env_gai_6_llm_add_ce_predictor_docker import ConstraintSimplificationEnv_test

# from test_code_bert_4 import CodeEmbedder, CodeEmbedder_normalize
from bert_embedder_test import CodeEmbedder_normalize
from test_rl.bert_predictor_2_mask import EnhancedEightClassModel
from test_rl.bert_predictor_mask import SimpleClassifier
from test_rl.test_script.utils import parse_smt2_in_parts, process_smt_lib_string, fetch_data_as_dict, \
    solve_and_measure_time, model_to_dict, load_dictionary, extract_variables_from_smt2_content, normalize_variables, \
    normalize_smt_str, MyException, timeout_handler, setup_logger
from test_rl.test_script.online_learning_break import online_learning

start = time.time()

def test_group():
    setup_logger()
    #pysmt方法
    # from pysmt.shortcuts import Symbol, Int, Equals, NotEquals, GT, LT, Plus, Times, Minus
    # from pysmt.typing import INT
    #
    # from pysmt.shortcuts import Symbol, Int, Equals, NotEquals, GT, LT, Plus, Times, Minus, And, write_smtlib
    # from pysmt.typing import INT
    #
    # # Declare integer symbols
    # x = Symbol('x', INT)
    # y = Symbol('y', INT)
    # z = Symbol('z', INT)
    # a = Symbol('a', INT)
    # b = Symbol('b', INT)
    # c = Symbol('c', INT)
    #
    # # Define custom Mod function using Python-like semantics
    # def Mod(a, b):
    #     return Minus(a, Times(Div(a, b), b))
    #
    # def Div(a, b):
    #     # Integer division that rounds toward zero
    #     return a.__truediv__(b)
    #
    # # Helper for power since pysmt Pow only supports reals; use repeated multiplication for integers
    # def ipow(base, exp):
    #     if exp == 0:
    #         return Int(1)
    #     result = base
    #     for _ in range(exp - 1):
    #         result = Times(result, base)
    #     return result
    #
    # # Build constraints list C1
    # C1 = []
    #
    # # 1st constraint:
    # expr1 = Plus(
    #     Mod(Plus(Times(x, Int(37)), Int(23)), Int(101)),
    #     Mod(Times(Minus(x, Int(30)), Minus(x, Int(30))), Int(57)),
    #     ipow(y, 3),
    #     Minus(Int(0), ipow(z, 2)),
    #     ipow(a, 2),
    #     Minus(Int(0), b),
    #     c
    # )
    # C1.append(Equals(expr1, Int(150)))
    #
    # # 2nd constraint:
    # expr2 = Plus(
    #     ipow(x, 4),
    #     ipow(y, 3),
    #     Times(Int(-5), z, a),
    #     b,
    #     Minus(Int(0), ipow(c, 2))
    # )
    # C1.append(LT(expr2, Int(400)))
    #
    # # 3rd constraint:
    # def complex_hash_2(v):
    #     return Plus(
    #         Mod(Plus(Times(v, Int(12)), Int(301)), Int(101)),
    #         Mod(Times(Minus(v, Int(15)), Minus(v, Int(30))), Int(57)),
    #         Mod(
    #             Times(Minus(v, Int(-15)), Minus(v, Int(30)), Minus(v, Int(11))),
    #             Int(9)
    #         )
    #     )
    #
    # expr3 = Plus(
    #     ipow(x, 2),
    #     Minus(Int(0), ipow(y, 2)),
    #     complex_hash_2(z),
    #     a,
    #     Minus(Int(0), Times(b, c))
    # )
    # expr3_rhs = complex_hash_2(a)
    # C1.append(GT(expr3, expr3_rhs))
    #
    # # 4th constraint:
    # def complex_hash(v):
    #     return Plus(
    #         Mod(Plus(Times(v, Int(37)), Int(23)), Int(101)),
    #         Mod(Times(Minus(v, Int(30)), Minus(v, Int(30))), Int(57))
    #     )
    #
    # expr4 = Plus(
    #     ipow(x, 2),
    #     Minus(Int(0), ipow(y, 2)),
    #     complex_hash(z),
    #     a,
    #     Minus(Int(0), Times(b, c))
    # )
    # C1.append(NotEquals(expr4, Int(55)))
    #
    # # Write to SMT-LIB file
    # write_smtlib(And(*C1), "constraints.smt2")
    # print("✅ SMT-LIB 文件已写入：constraints.smt2")
    #
    #
    #
    # with open("constraints.smt2", 'r') as file:
    #     smtlib_str = file.read()
    # assertions = parse_smt2_string(smtlib_str)
    # solver = Solver()
    # for a in assertions:
    #     solver.add(a)
    # 创建求解器
    # 创建求解器
    solver = Solver()

    # 定义变量
    x = Int('x')
    y = Int('y')
    z = Int('z')
    a = Int('a')
    b = Int('b')
    c = Int('c')

    # C1 约束组（所有 % 已展开为四则运算）
    C1 = [
        # 替换complex_hash(x)
        ((x * 37 + 23) % 101 + (x - 30) * (x - 30) % 57) + y ** 3 - z ** 2 + a ** 2 - b + c == 150,
        x ** 4 + y ** 3 - 5 * z * a + b - c ** 2 < 400,
        # 替换complex_hash_2(z)和complex_hash_2(a)
        x ** 2 - y ** 2 + ((z * 12 + 301) % 101 + ((z - 15) * (z - 30)) % 57 + (
                (z + 15) * (z - 30) * (z - 11)) % 9) + a - b * c >
        ((a * 12 + 301) % 101 + ((a - 15) * (a - 30)) % 57 + ((a + 15) * (a - 30) * (a - 11)) % 9),
        x ** 2 - y ** 2 +  # 这里不需要替换complex_hash(z)，因为它已经在上一个约束中被替换
        ((z * 37 + 23) % 101 + (z - 30) * (z - 30) % 57) + a - b * c != 55
    ]

    # 添加约束到求解器
    solver.add(And(C1))
    smtlib_str = solver.to_smt2()
    print (smtlib_str)
    smtlib_str = z3_smtlib_to_pysmt_compatible(smtlib_str)
    #
    # result, model, time_taken = solve_and_measure_time(solver, 1200000)
    #
    # print(result, time_taken, model)
    info_name = 'info_dict_gai_6_normal_0518_pre_example.txt'
    if not os.path.exists(info_name):
        # 文件不存在时，创建文件
        info_dict = {}
        with open(info_name, 'w') as file:
            json.dump(info_dict, file, indent=4)
        print(f'文件{info_name} 已创建。')
    else:
        info_dict = load_dictionary(info_name)
        print(f'文件已存在。')


    smtlib_str, var_dict, constant_list = normalize_smt_str(smtlib_str)
    # if len(var_dict) > 20:
    #     continue


    result_list = [var_dict,result,time_taken]
    # if list1[0] == "sat":
    #     result_list.append(list1[3])
    # else:
    #     result_list.append(None)


    # # 先取消求解，使用原始文件中的求解结果
    # timeout = 999999999
    # # timeout = 10
    # result, model, time_taken = solve_and_measure_time(solver, timeout)
    #
    # print(result, time_taken)
    # if result == 'unsat' or time_taken < 300:
    #     continue
    #
    # result_list = []
    # result_list.append(result,time_taken)
    # if model:
    #     result_list.append(model_to_dict(model))
    # else:
    #     result_list.append(None)
    # print(result_list[-1])


    file_path = ''
    start_time = time.time()

    signal.alarm(60*20)
    signal.signal(signal.SIGALRM, timeout_handler)
    try:
        logger.info(f'开始执行:')
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
                action_dim=1,
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
    except MyException as e:
        logger.info(f'执行结束: {file_path}')
        print("time out")


    signal.alarm(0)
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
    with open(info_name, 'w') as file:
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
