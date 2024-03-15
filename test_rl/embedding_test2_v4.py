import json
import re
import time

from z3 import *


from pearl.policy_learners.sequential_decision_making.soft_actor_critic import SoftActorCritic
# from pearl.replay_buffers.sequential_decision_making.bootstrap_replay_buffer import BootstrapReplayBuffer
from pearl.replay_buffers.sequential_decision_making.bootstrap_replay_buffer import FIFOOffPolicyReplayBuffer
from pearl.utils.functional_utils.experimentation.set_seed import set_seed
from pearl.action_representation_modules.identity_action_representation_module import IdentityActionRepresentationModule
from pearl.history_summarization_modules.lstm_history_summarization_module import LSTMHistorySummarizationModule
from pearl.utils.functional_utils.train_and_eval.online_learning import online_learning
from pearl.pearl_agent import PearlAgent

import torch
import matplotlib.pyplot as plt
import numpy as np
from env import ConstraintSimplificationEnv_v3


from test_code_bert_4 import CodeEmbedder
from test_rl.test_script.utils import parse_smt2_in_parts, process_smt_lib_string

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
# predictor = Predictor('KNN')

# file_path = '/home/lz/baidudisk/smt/gnu_angr.tar.gz/single_test/arch/arch15998'
# file_path = '/home/nju/Downloads/smt/buzybox_angr.tar.gz/single_test/readahead/readahead651389'
file_path = '/home/nju/Downloads/smt/gnu_angr.tar.gz/single_test/seq/seq143541'
with open('time.txt', "a") as file:
    file.write(f"当前测试文件:{file_path}\n")
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
# assertions = process_smt_lib_string(smtlib_str)
variables = set()

# solver = Solver()
# for a in assertions:
#     solver.add(a)

# Extract variables from each assertion
# for a in assertions:
#     visit(a)
variables = extract_variables_from_smt2_content(smtlib_str)

# Print all variables
print("变量列表：")
for v in variables:
    print(v)

embedder = CodeEmbedder()


set_seed(0)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# model = SequenceClassificationModel(100).to(device)
# model.load_state_dict(torch.load("/home/lz/PycharmProjects/Pearl/pearl/tutorials/single_item_recommender_system_example/env_model_state_dict.pt"))
# actions = torch.load("/home/lz/PycharmProjects/Pearl/pearl/tutorials/single_item_recommender_system_example/news_embedding_small.pt")
# env = RecEnv(list(actions.values())[:100], model)
# observation, action_space = env.reset()

# 创建环境实例
# 创建环境实例
env = ConstraintSimplificationEnv_v3(embedder, assertions, len(variables), len(variables),smtlib_str)
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
number_of_steps = 100000
record_period = 400
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
        action_dim=len(env.variables)+1,
        hidden_dim=768,
        history_length=len(env.variables),  # 和完整结点数相同
    ),
    replay_buffer = FIFOOffPolicyReplayBuffer(10),
    device_id=-1,
)
# 训练代理
info = online_learning(
    agent=agent,
    env=env,
    number_of_steps=number_of_steps,
    print_every_x_steps=100,
    record_period=record_period,
    learn_after_episode=True,
)

torch.save(info["return"], "BootstrappedDQN-LSTM-return.pt")
plt.plot(record_period * np.arange(len(info["return"])), info["return"], label="BootstrappedDQN-LSTM")
plt.legend()
plt.show()

