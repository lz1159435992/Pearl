# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# pyre-ignore-all-errors
import json
import re
import copy
import torch
import z3
from torch.nn.parameter import Parameter
from z3 import *
import embedding_util
from pearl.SMTimer.KNN_Predictor import Predictor
from test_rl.test_script.db_search_lz_alue import fetch_data_as_dict
from test_rl.test_script.utils import find_var_declaration_in_string, split_at_check_sat, load_dictionary, \
    find_assertions_related_to_var_name, find_assertions_related_to_var_names_optimized, \
    find_assertions_related_to_var_names_optimized_dfs

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0")

import torch
import numpy as np
import random
from pearl.api.action_result import ActionResult
from pearl.api.environment import Environment
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace
import datetime


def update_txt_with_current_time(file_path, solve_time):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(file_path, "a") as file:
        file.write(f"current_time:{current_time}\n" + f"file:" + f"solve_time:{solve_time}s\n")


sys.path.append('/home/nju/PycharmProjects/Pearl/test_rl')
NODE_TYPE_ENUM = {
    "Variable-Int": 0,  # 布尔表达式
    "Variable-Real": 1,  # 算术表达式
    "Constant": 2,  # 变量
    "BoolExpr": 3,  # 常量
    "ArithExpr": 4,  # 量词
    "Quantifier": 5,  # 函数和关系
    "Operation": 6,  # 函数和关系
    "Unknown": 7  # 函数和关系
}
EDGE_TYPE_ENUM = {
    "ParentChild": 0,
    "Sibling": 1,
    # 根据需求可以添加更多边的类型
}


def get_values(db_path, table_name):
    # db_path = 'value_dictionary.db'
    # table_name = 'value_dictionary'
    value_dict_1 = fetch_data_as_dict(db_path, table_name)
    value_dict_2 = {}
    count = 0
    for key in value_dict_1.keys():
        if int(value_dict_1[key]) > 1:
            value_dict_2[str(count)] = key
            count += 1
    return value_dict_2


def get_values_nju(db_path, table_name):
    # db_path = 'value_dictionary.db'
    # table_name = 'value_dictionary'
    value_dict_2 = {}
    value_dict_1 = fetch_data_as_dict(db_path, table_name)
    sampled_items = random.sample(value_dict_1.items(), 20000)
    value_dict_1 = dict(sampled_items)
    count = 0
    for key in value_dict_1.keys():
        value_dict_2[str(count)] = key
        count += 1
    value_dict_2[str(count)] = 'random'
    return value_dict_2


def group_values(input_dict, group_size):
    # 确保group_size是正数
    if group_size <= 0:
        raise ValueError("group_size must be a positive integer")

    # 将字典的值转换为整数并进行排序
    sorted_values = []
    for value in input_dict.values():
        try:
            sorted_values.append(int(value))
        except ValueError:
            # 跳过无法转换为整数的值
            continue

    sorted_values.sort()

    # 初始化结果字典
    result_dict = {}

    # 使用切片将排序后的列表分组
    for i in range(0, len(sorted_values), group_size):
        result_dict[i // group_size] = sorted_values[i:i + group_size]

    return result_dict


# duqu shuzhi zidian
# with open('test_script/dict_value.txt', 'r') as value_file:
#     # 璇诲彇鏂囦欢鎵€鏈夊唴瀹瑰埌涓€涓瓧绗︿覆
#     value_str = value_file.read()
# try:
#     # 灏咼SON瀛楃涓茶浆鎹负瀛楀吀
#     dict_value = json.loads(value_str)
#     print(dict_value)
#     # print("杞崲鍚庣殑瀛楀吀锛?, dict_obj)
# except json.JSONDecodeError as e:
#     print('failed', e)
# nju
dict_value = get_values_nju('/home/lz/PycharmProjects/Pearl/test_rl/test_script/value_dictionary.db',
                            'value_dictionary')


# fenzu
# dict_value = group_values(dict_value,100)
class CustomEnvironment(Environment):
    def __init__(self, model, encoder, decoder, graph_builder, num_vars, num_consts):
        self.model = model.to(device)
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.graph_builder = graph_builder
        self.num_vars = num_vars
        self.num_consts = num_consts
        self.T = 20  # 最大步数
        self.t = 0  # 当前步数

    def reset(self, smtlib_str, seed=None):
        self.t = 0
        # 使用graph_builder从smtlib字符串构建图
        graph = self.graph_builder.build_graph_from_smtlib(smtlib_str)
        # 使用encoder获取节点嵌入
        node_embeddings = self.encoder(graph)
        # 将node_embeddings传递给decoder，以预测变量和常数
        # self.variable_preds, self.constant_preds = self.decoder(node_embeddings)
        self.action_space = DiscreteActionSpace(list(range(self.num_vars * self.num_consts)))
        return [0.0], self.action_space

    def step(self, action):
        # 将动作转换为变量和常数的选择
        var_index = action // self.num_consts
        const_index = action % self.num_consts
        chosen_var = self.variable_preds[var_index]
        chosen_const = self.constant_preds[const_index]

        # 假设model可以使用chosen_var和chosen_const进行计算并返回奖励
        reward = self.model(chosen_var, chosen_const)
        true_reward = np.random.binomial(1, reward)

        self.t += 1
        terminated = self.t >= self.T
        return ActionResult(
            observation=[float(true_reward)],
            reward=float(true_reward),
            terminated=terminated,
            truncated=False,
            info={},
            available_action_space=self.action_space
        )


class ConstraintSimplificationEnv(Environment):

    def __init__(self, encoder, decoder, z3ast, num_variables, num_constants):
        self.decoder = decoder
        self.encoder = encoder
        self.z3ast = z3ast
        self.z3ast_original = z3ast
        self.num_variables = num_variables
        self.num_constants = num_constants
        self.state = None
        self.variables = set()
        self.actions = []
        self.concrete_finish = False
        self.concrete_count = 0
        self.counterexamples_list = []
        self.finish = False
        self.used_variables = []
        # 记录state输入了多少次
        self.state_count = 0
        self.predictor = Predictor('KNN')

    def reset(self, seed=None):
        self.concrete_finish = False
        self.concrete_count = 0
        self.finish = False
        self.used_variables = []
        # 从原始的ast开始构建s

        graph = embedding_util.Z3ASTGraph(self.z3ast_original)
        # node_type_dict = NODE_TYPE_ENUM
        graph2vec = embedding_util.Graph2Vec(graph)
        # 步骤5: 输出转换结果
        print("节点特征向量:")
        print(graph2vec.node_feat.shape)
        # node_embed = embedding_util.glorot_uniform(graph2vec.node_feat)
        node_embed = Parameter(graph2vec.node_feat)
        self.state = self.encoder(node_embed)
        variables = set()
        for a in self.z3ast:
            visit(a, variables)
        self.variables = list(variables)
        # 之后要修改成变量+常量
        # for i in range(len(self.variables)):
        #     self.actions.append(torch.tensor(i))
        #     # 先不使用字典了
        self.actions = self.strings_to_onehot(self.variables)
        self.actions.to(device)
        print('++++++++++++++++++++')
        print(self.actions)
        print(self.actions.shape)
        # self.variables = {index: item for index, item in enumerate(self.variables)}
        self.action_space = DiscreteActionSpace(self.actions)
        return self.state, self.action_space

    def action_space(self):
        """Returns the action space of the environment."""
        pass

    def step(self, action):
        reward = 0
        # variable_pred = self.variables[action]
        action = self.action_space.actions_batch[action]
        print('////////////////////////////')
        print(action)
        print(type(action))
        print(self.onehot_to_indices(action))
        variable_pred = self.variables[self.onehot_to_indices(action)[0]]
        # 在一次执行过程中，action不能重复
        if self.concrete_count == 0:
            self.counterexamples_list.append([])
        if variable_pred not in self.used_variables:
            self.used_variables.append(variable_pred)
            self.concrete_count += 1
            # 数值这部分需要修改
            min_int32 = -2147483648
            max_int32 = 2147483647

            # 生成一个随机的32位整数
            random_int = random.randint(min_int32, max_int32)

            self.counterexamples_list[-1].append([variable_pred, random_int])

            solver = Solver()
            for a in self.z3ast:
                solver.add(a)
            exec(f"{variable_pred} = Int('{variable_pred}')")
            # 修改，添加取值部分内容

            solver.add(eval(variable_pred) == random_int)
            reward += self.calculate_reward(solver)
            self.z3ast = solver.assertions()
            graph = embedding_util.Z3ASTGraph(self.z3ast)
            # node_type_dict = NODE_TYPE_ENUM
            graph2vec = embedding_util.Graph2Vec(graph)
            # 步骤5: 输出转换结果
            print("节点特征向量:")
            print(graph2vec.node_feat.shape)
            # node_embed = embedding_util.glorot_uniform(graph2vec.node_feat)
            node_embed = Parameter(graph2vec.node_feat)
            self.state = self.encoder(node_embed)

            if self.concrete_count == len(self.variables):
                self.concrete_finish = True
                self.reset()
                # 判断这里需不需要直接reset
        else:
            reward += -10
            print(action)
            # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            device = torch.device("cpu")
            self.actions = [act.to(device) for act in self.actions]
            print(self.actions)
            for i in self.actions:
                i.to(device)
            action.to(device)
            self.actions = [tensor1 for tensor1 in self.actions if
                            not any(torch.equal(tensor1, tensor2) for tensor2 in action)]
            self.action_space = DiscreteActionSpace(self.actions)
        return ActionResult(
            observation=self.state,
            reward=float(reward),
            terminated=self.concrete_finish,
            truncated=self.finish,
            info={},
            available_action_space=self.action_space, )

    @staticmethod
    def strings_to_onehot(string_list):
        # 创建一个从字符串到索引的映射
        str_to_index = {string: index for index, string in enumerate(string_list)}

        # 创建One-Hot编码的张量
        one_hot_tensors = []
        for string in string_list:
            # 创建一个全0的向量
            one_hot_vector = torch.zeros(len(string_list), dtype=torch.float32)
            # 将对应位置置1
            one_hot_vector[str_to_index[string]] = 1.0
            one_hot_vector.to(device)
            one_hot_tensors.append(one_hot_vector)
        one_hot_matrix = torch.stack(one_hot_tensors)

        return one_hot_matrix
        # return one_hot_tensors

    @staticmethod
    def onehot_to_indices(one_hot_tensors):
        # 将One-Hot编码的张量转换回索引
        return [torch.argmax(tensor).item() for tensor in one_hot_tensors]

    @staticmethod
    def counter_reward_function(total_length, unique_count):

        """
        Calculate the reward based on the total length of the list and the number of unique in it.

        Args:
        - total_length (int): The total length of the list.
        - unique_count (int): The number of unique in the list.

        Returns:
        - float: The calculated reward.
        """
        # Define the base reward values
        R_positive = 1
        R_negative = -1

        # Define the scaling factor for negative reward
        alpha = 1 / math.sqrt(total_length) if total_length > 0 else 1

        # Check if there are any unique strings
        if unique_count > 0:
            # Calculate the positive reward, scaled based on the list length
            reward = R_positive / math.log(1 + total_length)
        else:
            # Apply the negative reward, scaled by alpha
            reward = R_negative * alpha

        return reward

    def calculate_reward(self, solver):
        reward = 0
        count = 0
        solver.set("timeout", 60000)
        # 判断新产生的序列和之前有没有重复
        # 判断是否存在反例
        if len(self.counterexamples_list) > 1:
            if self.counterexamples_list[-1] in self.counterexamples_list[:len(self.counterexamples_list) - 1]:
                reward += -1
            else:
                # 判断新的序列和之前是否有重复（字符串重复）
                # for i in range(len(self.counterexamples_list) - 1):
                #     # if self.are_lists_equal(self.counterexamples_list[i],self.counterexamples_list[-1]):
                #     if ' '.join(self.counterexamples_list[-1]) in ' '.join(self.counterexamples_list[i]):
                #         count += 1
                last_joined = ' '.join(
                    ' '.join(str(item) for item in inner_list) for inner_list in self.counterexamples_list[-1])
                for i in range(len(self.counterexamples_list) - 1):
                    current_joined = ' '.join(
                        ' '.join(str(item) for item in inner_list) for inner_list in self.counterexamples_list[i])
                    if last_joined in current_joined:
                        count += 1
                reward += self.counter_reward_function(len(self.counterexamples_list) - 1,
                                                       len(self.counterexamples_list) - 1 - count)
            print(self.counterexamples_list)
        # 后续实现一些子集求解
        query_smt2 = solver.to_smt2()
        # print(query_smt2)
        predicted_solvability = self.predictor.predict(query_smt2)
        if predicted_solvability == 0:
            # 提高一下reward数值
            reward += 2
            r = solver.check()
            stats = solver.statistics()
            if z3.sat == r:

                self.finish = True

                print("求解时间:", stats.get_key_value('time'))
            else:
                # reward += 1 / stats.get_key_value('time') * 100
                reward += -5

        return reward

    def are_lists_equal(self, list1, list2):
        if len(list1) != len(list2):
            return False

        for item1, item2 in zip(list1, list2):
            if item1 != item2:
                return False

        return True

    def render(self) -> None:
        """Renders the environment. Default implementation does nothing."""
        return None

    def close(self) -> None:
        """
        Closes environment, taking care of any cleanup needed.
        Default implementation does nothing.
        """
        return None


class ConstraintSimplificationEnv_v2(Environment):

    def __init__(self, encoder, decoder, z3ast, num_variables, num_constants):
        self.actions_v = None
        self.decoder = decoder
        self.encoder = encoder
        self.z3ast = z3ast
        self.z3ast_original = z3ast
        self.num_variables = num_variables
        self.num_constants = num_constants
        self.state = None
        self.variables = set()
        self.actions = []
        self.concrete_finish = False
        self.concrete_count = 0
        self.counterexamples_list = []
        self.finish = False
        self.used_variables = []
        # 记录state输入了多少次
        self.state_count = 0
        self.predictor = Predictor('KNN')

    def reset(self, seed=None):
        self.concrete_finish = False
        self.concrete_count = 0
        self.finish = False
        self.used_variables = []
        # 从原始的ast开始构建s

        graph = embedding_util.Z3ASTGraph(self.z3ast_original)
        # node_type_dict = NODE_TYPE_ENUM
        graph2vec = embedding_util.Graph2Vec(graph)
        # 步骤5: 输出转换结果
        # node_embed = embedding_util.glorot_uniform(graph2vec.node_feat)
        node_embed = Parameter(graph2vec.node_feat)
        self.state = self.encoder(node_embed)
        variables = set()
        for a in self.z3ast:
            visit(a, variables)
        self.variables = list(variables)
        # 之后要修改成变量+常量
        # for i in range(len(self.variables)):
        #     self.actions.append(torch.tensor(i))
        #     # 先不使用字典了
        # tensor = torch.arange(-10000, 10001)
        # 笛卡尔积
        self.actions_v = self.strings_to_onehot(self.variables)
        self.actions = get_actions(self.actions_v, torch.arange(-10000, 10001))

        self.actions.to(device)
        # self.variables = {index: item for index, item in enumerate(self.variables)}
        # self.action_space = BoxActionSpace([torch.tensor(0), torch.tensor(-10000)],
        #                                    [torch.tensor(len(self.variables)), torch.tensor(10000)])
        self.action_space = DiscreteActionSpace(self.actions)
        return self.state, self.action_space

    def action_space(self):
        """Returns the action space of the environment."""
        pass

    def step(self, action):
        reward = 0
        # variable_pred = self.variables[action]
        # action = self.action_space.
        action = self.action_space.actions_batch[action]
        action_v = action[:-1]
        action_n = action[-1]
        variable_pred = self.variables[self.onehot_to_indices(action_v)[0]]
        # 在一次执行过程中，action不能重复
        if self.concrete_count == 0:
            self.counterexamples_list.append([])
        if variable_pred not in self.used_variables:
            self.used_variables.append(variable_pred)
            self.concrete_count += 1
            # 数值这部分需要修改
            # min_int32 = -2147483648
            # max_int32 = 2147483647
            #
            # # 生成一个随机的32位整数
            # random_int = random.randint(min_int32, max_int32)
            selected_int = action_n.item()
            self.counterexamples_list[-1].append([variable_pred, selected_int])

            solver = Solver()
            for a in self.z3ast:
                solver.add(a)
            exec(f"{variable_pred} = Int('{variable_pred}')")
            # 修改，添加取值部分内容

            solver.add(eval(variable_pred) == selected_int)
            reward += self.calculate_reward(solver)
            self.z3ast = solver.assertions()
            graph = embedding_util.Z3ASTGraph(self.z3ast)
            # node_type_dict = NODE_TYPE_ENUM
            graph2vec = embedding_util.Graph2Vec(graph)
            # 步骤5: 输出转换结果
            print("节点特征向量:")
            print(graph2vec.node_feat.shape)
            # node_embed = embedding_util.glorot_uniform(graph2vec.node_feat)
            node_embed = Parameter(graph2vec.node_feat)
            self.state = self.encoder(node_embed)

            if self.concrete_count == len(self.variables):
                self.concrete_finish = True
                self.reset()
                # 判断这里需不需要直接reset
        else:
            reward += -10
            print(action)
            # 重新实现
            # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            device = torch.device("cpu")
            self.actions_v = [act.to(device) for act in self.actions_v]
            action_v = [act.to(device) for act in action_v]
            print(self.actions)
            for i in self.actions_v:
                i.to(device)
            for i in action_v:
                i.to(device)
            # action_v.to(device)
            self.actions_v = [tensor1 for tensor1 in self.actions_v if
                              not any(torch.equal(tensor1, tensor2) for tensor2 in action_v)]
            self.action_space = DiscreteActionSpace(get_actions(self.actions_v, torch.arange(-10000, 10001)))
        # 清除内存
        torch.cuda.empty_cache()
        return ActionResult(
            observation=self.state,
            reward=float(reward),
            terminated=self.concrete_finish,
            truncated=self.finish,
            info={},
            available_action_space=self.action_space, )

    @staticmethod
    def strings_to_onehot(string_list):
        # 创建一个从字符串到索引的映射
        str_to_index = {string: index for index, string in enumerate(string_list)}

        # 创建One-Hot编码的张量
        one_hot_tensors = []
        for string in string_list:
            # 创建一个全0的向量
            one_hot_vector = torch.zeros(len(string_list), dtype=torch.float32)
            # 将对应位置置1
            one_hot_vector[str_to_index[string]] = 1.0
            one_hot_vector.to(device)
            one_hot_tensors.append(one_hot_vector)
        one_hot_matrix = torch.stack(one_hot_tensors)

        return one_hot_matrix
        # return one_hot_tensors

    @staticmethod
    def onehot_to_indices(one_hot_tensors):
        # 将One-Hot编码的张量转换回索引
        return [torch.argmax(tensor).item() for tensor in one_hot_tensors]

    @staticmethod
    def counter_reward_function(total_length, unique_count):

        """
        Calculate the reward based on the total length of the list and the number of unique in it.

        Args:
        - total_length (int): The total length of the list.
        - unique_count (int): The number of unique in the list.

        Returns:
        - float: The calculated reward.
        """
        # Define the base reward values
        R_positive = 1
        R_negative = -1

        # Define the scaling factor for negative reward
        alpha = 1 / math.sqrt(total_length) if total_length > 0 else 1

        # Check if there are any unique strings
        if unique_count > 0:
            # Calculate the positive reward, scaled based on the list length
            reward = R_positive / math.log(1 + total_length)
        else:
            # Apply the negative reward, scaled by alpha
            reward = R_negative * alpha

        return reward

    def calculate_reward(self, solver):
        reward = 0
        count = 0
        solver.set("timeout", 60000)
        # 判断新产生的序列和之前有没有重复
        # 判断是否存在反例
        if len(self.counterexamples_list) > 1:
            if self.counterexamples_list[-1] in self.counterexamples_list[:len(self.counterexamples_list) - 1]:
                reward += -1
            else:
                # 判断新的序列和之前是否有重复（字符串重复）
                # for i in range(len(self.counterexamples_list) - 1):
                #     # if self.are_lists_equal(self.counterexamples_list[i],self.counterexamples_list[-1]):
                #     if ' '.join(self.counterexamples_list[-1]) in ' '.join(self.counterexamples_list[i]):
                #         count += 1
                last_joined = ' '.join(
                    ' '.join(str(item) for item in inner_list) for inner_list in self.counterexamples_list[-1])
                for i in range(len(self.counterexamples_list) - 1):
                    current_joined = ' '.join(
                        ' '.join(str(item) for item in inner_list) for inner_list in self.counterexamples_list[i])
                    if last_joined in current_joined:
                        count += 1
                reward += self.counter_reward_function(len(self.counterexamples_list) - 1,
                                                       len(self.counterexamples_list) - 1 - count)
            print(self.counterexamples_list)
        # 后续实现一些子集求解
        query_smt2 = solver.to_smt2()
        # print(query_smt2)
        predicted_solvability = self.predictor.predict(query_smt2)
        if predicted_solvability == 0:
            # 提高一下reward数值
            reward += 2
            r = solver.check()
            stats = solver.statistics()
            if z3.sat == r:

                self.finish = True

                print("求解时间:", stats.get_key_value('time'))
            else:
                # reward += 1 / stats.get_key_value('time') * 100
                reward += -5

        return reward

    def are_lists_equal(self, list1, list2):
        if len(list1) != len(list2):
            return False

        for item1, item2 in zip(list1, list2):
            if item1 != item2:
                return False

        return True

    def render(self) -> None:
        """Renders the environment. Default implementation does nothing."""
        return None

    def close(self) -> None:
        """
        Closes environment, taking care of any cleanup needed.
        Default implementation does nothing.
        """
        return None


class ConstraintSimplificationEnv_test(Environment):

    def __init__(self, embedder, z3ast, num_variables, num_constants, smtlib_str, file_path):
        self.step_count = 0
        self.file_path = file_path
        self.actions_v = None
        self.embedder = embedder
        self.z3ast = z3ast
        self.z3ast_original = copy.deepcopy(z3ast)
        self.num_variables = num_variables
        self.num_constants = num_constants
        self.smtlib_str = smtlib_str
        self.smtlib_str_original = copy.deepcopy(smtlib_str)
        self.state_original = self.embedder.get_max_pooling_embedding(self.smtlib_str)
        self.state = None
        self.variables = extract_variables_from_smt2_content(self.smtlib_str)
        self.actions = []
        self.concrete_finish = False
        self.concrete_count = 0
        self.counterexamples_list = []
        self.finish = False
        self.used_variables = []
        # 记录state输入了多少次
        self.state_count = 0
        self.predictor = Predictor('KNN')
        self.last_performance = 0
        self.solve_time = 0
        self.v_related_assertions = find_assertions_related_to_var_names_optimized_dfs(self.z3ast, self.variables)

    def reset(self, seed=None):
        self.concrete_finish = False
        self.concrete_count = 0
        # self.finish = False
        self.used_variables = []
        # 从原始的ast开始构建s
        self.state = copy.deepcopy(self.state_original)
        # self.state = self.embedder.get_max_pooling_embedding(self.smtlib_str)
        self.z3ast = copy.deepcopy(self.z3ast_original)
        self.smtlib_str = copy.deepcopy(self.smtlib_str_original)
        self.last_performance = 0
        self.actions_v = self.strings_to_onehot(self.variables)

        self.actions = get_actions(self.actions_v, torch.arange(0, len(dict_value) - 1))

        self.actions.to(device)
        # self.variables = {index: item for index, item in enumerate(self.variables)}
        # self.action_space = BoxActionSpace([torch.tensor(0), torch.tensor(-10000)],
        #                                    [torch.tensor(len(self.variables)), torch.tensor(10000)])
        self.action_space = DiscreteActionSpace(self.actions)
        del self.actions
        torch.cuda.empty_cache()
        return self.state, self.action_space

    def action_space(self):
        """Returns the action space of the environment."""
        pass

    def step(self, action):
        # print(action)
        self.step_count += 1
        try:
            reward = 0
            # variable_pred = self.variables[action]
            # action = self.action_space.
            action = self.action_space.actions_batch[action]
            action_v = action[:-1]
            action_n = action[-1]
            # print(self.onehot_to_indices(action_v))
            variable_pred = self.variables[self.onehot_to_indices(action_v)]
            # print(self.counterexamples_list)
            print(variable_pred)
            # 在一次执行过程中，action不能重复
            if self.concrete_count == 0:
                self.counterexamples_list.append([])
            if variable_pred not in self.used_variables:
                # 修改的部分
                type_info = find_var_declaration_in_string(self.smtlib_str_original, variable_pred)
                print(type_info)
                print(type(type_info))
                type_scale = type_info.split(' ')[-1]
                print(type_scale)
                max_value = 2 ** int(type_scale) - 1
                min_value = -2 ** (int(type_scale) - 1)
                # 如果选择了最后一个数，随机选择一个值
                if str(dict_value[str(int(action_n.item()))]) == 'random':
                    selected_int = random.randrange(min_value, max_value + 1)
                else:
                    selected_int = int(dict_value[str(int(action_n.item()))])
                # 找到最大最小值
                if min_value <= selected_int <= max_value:
                    reward += 5
                    # 需要添加的约束
                    new_constraint = "(assert (= {} (_ bv{} {})))\n".format(variable_pred, selected_int, type_scale)

                    # assertions = parse_smt2_string(self.smtlib_str)
                    related_assertions = self.v_related_assertions[variable_pred]
                    if len(related_assertions) > 0:
                        # related_assertions = find_assertions_related_to_var_name(assertions, variable_pred)
                        solver_related = Solver()

                        for a in related_assertions:
                            solver_related.add(a)
                        # 先预测再求解
                        smtlib_str_before, smtlib_str_after = split_at_check_sat(solver_related.to_smt2())

                        new_smtlib_str = smtlib_str_before + new_constraint + smtlib_str_after
                        predicted_solvability_related = self.predictor.predict(new_smtlib_str)
                        # 统计求解正确率
                        if not os.path.exists('predict.json'):
                            # 文件不存在时，创建文件
                            pre_list = []
                            with open('predict.json', 'w') as file:
                                json.dump(pre_list, file, indent=4)
                            print(f"文件 predict.txt.txt 已创建。")
                        else:
                            with open('predict.json', 'r') as f:
                                pre_list = json.load(f)
                        temp_list = []

                        if predicted_solvability_related == 0:
                            temp_list.append(0)
                            reward += 5
                            assertions_related = parse_smt2_string(new_smtlib_str)
                            solver_related = Solver()
                            for a in assertions_related:
                                solver_related.add(a)
                            solver_related.set("timeout", 600000)
                            r = solver_related.check()
                            if z3.sat == r:
                                temp_list.append(0)
                                reward += 5
                                self.used_variables.append(variable_pred)
                                self.concrete_count += 1
                                # 数值这部分需要修改
                                # print(action_n.item)
                                # print(type(action_n.item)
                            elif z3.unknown == r:
                                temp_list.append('unknown')
                                reward += 2
                                self.used_variables.append(variable_pred)
                                self.concrete_count += 1
                                # 数值这部分需要修改
                                # print(action_n.item)
                                # print(type(action_n.item))

                            else:
                                reward += -5

                                temp_list.append(1)

                                pre_list.append(temp_list)
                                with open('predict.json', 'w') as file:
                                    json.dump(pre_list, file, indent=4)

                            print(selected_int)
                            self.counterexamples_list[-1].append([variable_pred, selected_int])

                            smtlib_str_before, smtlib_str_after = split_at_check_sat(self.smtlib_str)
                            # new_constraint = "(assert (= {} (_ bv{} {})))\n".format(variable_pred, selected_int, type_scale)
                            self.smtlib_str = smtlib_str_before + new_constraint + smtlib_str_after
                            assertions = parse_smt2_string(self.smtlib_str)
                            solver = Solver()
                            for a in assertions:
                                solver.add(a)
                            reward += self.calculate_reward(solver)
                            self.z3ast = solver.assertions()
                            self.state = self.embedder.get_max_pooling_embedding(solver.to_smt2())

                            if self.concrete_count == len(self.variables):
                                self.concrete_finish = True
                                self.reset()
                            self.actions_v = [act.to(device) for act in self.actions_v]
                            self.actions_v = [torch.round(act) for act in self.actions_v]
                            action_v = [act.to(device) for act in action_v]
                            action_v = [torch.round(act) for act in action_v]
                            self.actions_v = [tensor1 for tensor1 in self.actions_v if
                                              not any(torch.equal(tensor1, tensor2) for tensor2 in action_v)]
                            self.action_space = DiscreteActionSpace(
                                get_actions(self.actions_v, torch.arange(0, len(dict_value) - 1)))
                        else:
                            reward += -5
                    else:  # 没有变量约束的情况
                        reward += 0
                        self.used_variables.append(variable_pred)
                        self.concrete_count += 1
                        # 数值这部分需要修改
                        # print(action_n.item)
                        # print(type(action_n.item))

                        print(selected_int)
                        self.counterexamples_list[-1].append([variable_pred, selected_int])

                        smtlib_str_before, smtlib_str_after = split_at_check_sat(self.smtlib_str)
                        # new_constraint = "(assert (= {} (_ bv{} {})))\n".format(variable_pred, selected_int, type_scale)
                        self.smtlib_str = smtlib_str_before + new_constraint + smtlib_str_after
                        assertions = parse_smt2_string(self.smtlib_str)
                        solver = Solver()
                        for a in assertions:
                            solver.add(a)
                        reward += self.calculate_reward(solver)
                        self.z3ast = solver.assertions()
                        self.state = self.embedder.get_max_pooling_embedding(solver.to_smt2())

                        self.actions_v = [act.to(device) for act in self.actions_v]
                        self.actions_v = [torch.round(act) for act in self.actions_v]
                        action_v = [act.to(device) for act in action_v]
                        action_v = [torch.round(act) for act in action_v]
                        self.actions_v = [tensor1 for tensor1 in self.actions_v if
                                          not any(torch.equal(tensor1, tensor2) for tensor2 in action_v)]
                        self.action_space = DiscreteActionSpace(
                            get_actions(self.actions_v, torch.arange(0, len(dict_value) - 1)))

                        if self.concrete_count == len(self.variables):
                            self.concrete_finish = True
                            self.reset()
                else:
                    reward += -5
            else:
                reward += -10
                print(action)
            # print('***********************')
            # print(len(self.counterexamples_list))
            # for i in self.counterexamples_list:
            #     print(len(i))
            # 清除内存
            del action
            del action_n
            del action_v
            torch.cuda.empty_cache()
        except Exception as e:
            print('some problems are triggered')
            self.state = copy.deepcopy(self.state_original)
            reward = 0
        if self.step_count > 500:
            self.finish = True
        return ActionResult(
            observation=self.state,
            reward=float(reward),
            terminated=self.finish,
            truncated=False,
            info={},
            available_action_space=self.action_space, )

    @staticmethod
    def strings_to_onehot(string_list):
        # 创建一个从字符串到索引的映射
        str_to_index = {string: index for index, string in enumerate(string_list)}

        # 创建One-Hot编码的张量
        one_hot_tensors = []
        for string in string_list:
            # 创建一个全0的向量
            one_hot_vector = torch.zeros(len(string_list), dtype=torch.float32)
            # 将对应位置置1
            one_hot_vector[str_to_index[string]] = 1.0
            one_hot_vector.to(device)
            one_hot_tensors.append(one_hot_vector)
        one_hot_matrix = torch.stack(one_hot_tensors)
        del one_hot_vector
        del one_hot_tensors
        torch.cuda.empty_cache()
        return one_hot_matrix
        # return one_hot_tensors

    @staticmethod
    def onehot_to_indices(one_hot_tensors):
        # 将One-Hot编码的张量转换回索引
        return torch.argmax(one_hot_tensors).item()

    @staticmethod
    def counter_reward_function(total_length, unique_count):

        """
        Calculate the reward based on the total length of the list and the number of unique in it.

        Args:
        - total_length (int): The total length of the list.
        - unique_count (int): The number of unique in the list.

        Returns:
        - float: The calculated reward.
        """
        # Define the base reward values
        R_positive = 1
        R_negative = -1

        # Define the scaling factor for negative reward
        alpha = 1 / math.sqrt(total_length) if total_length > 0 else 1

        # Check if there are any unique strings
        if unique_count > 0:
            # Calculate the positive reward, scaled based on the list length
            reward = R_positive / math.log(1 + total_length) * 10
        else:
            # Apply the negative reward, scaled by alpha
            reward = R_negative * alpha * 10

        return reward

    def calculate_reward(self, solver):
        performance = 0
        reward = 0
        count = 0
        solver.set("timeout", 60000)
        # 判断新产生的序列和之前有没有重复
        # 判断是否存在反例
        if len(self.counterexamples_list) > 1:
            if self.counterexamples_list[-1] in self.counterexamples_list[:len(self.counterexamples_list) - 1]:
                reward += -10
            else:
                # 判断新的序列和之前是否有重复（字符串重复）
                # for i in range(len(self.counterexamples_list) - 1):
                #     # if self.are_lists_equal(self.counterexamples_list[i],self.counterexamples_list[-1]):
                #     if ' '.join(self.counterexamples_list[-1]) in ' '.join(self.counterexamples_list[i]):
                #         count += 1
                last_joined = ' '.join(
                    ' '.join(str(item) for item in inner_list) for inner_list in self.counterexamples_list[-1])
                for i in range(len(self.counterexamples_list) - 1):
                    current_joined = ' '.join(
                        ' '.join(str(item) for item in inner_list) for inner_list in self.counterexamples_list[i])
                    if last_joined in current_joined:
                        count += 1
                reward += self.counter_reward_function(len(self.counterexamples_list) - 1,
                                                       len(self.counterexamples_list) - 1 - count)
                # print(self.counterexamples_list)
                # print(len(self.counterexamples_list))
                # for i in self.counterexamples_list:
                #     print(len(i))
                # 后续实现一些子集求解
                # 注释掉提高速度
                solver_part = Solver()
                assertions = solver.assertions()

                assertions_list = []
                for a in assertions:
                    assertions_list.append(a)

                indexes = random.sample(range(len(assertions_list)), int(len(assertions) * 0.5))

                # 根据索引列表，从原始列表中选取元素，并保持原始顺序
                res = [assertions_list[i] for i in sorted(indexes)]
                # res = random.sample(assertions_list, int(len(assertions) * 0.6))
                for r in res:
                    solver_part.add(r)
                predicted_solvability_part = self.predictor.predict(solver_part.to_smt2())
                if predicted_solvability_part == 0:
                    # if True:
                    performance += 1
                    reward += 10
                    # 注释掉提高速度
                    solver_part.set("timeout", 30000)
                    # r = solver_part.check()
                    # if z3.sat == r:
                    if True:
                        #     performance += 1
                        #     reward += 10
                        predicted_solvability = self.predictor.predict(self.smtlib_str)
                        if predicted_solvability == 0:
                            performance += 1
                            # 提高一下reward数值
                            reward += 15
                            r = solver.check()
                            stats = solver.statistics()
                            if z3.sat == r:
                                performance += 1
                                reward += 20
                                self.finish = True
                                self.solve_time = stats.get_key_value('time')
                                print("求解时间:", stats.get_key_value('time'))
                            else:
                                # reward += 1 / stats.get_key_value('time') * 100
                                reward += -20
                        else:
                            reward += -15
                    elif z3.sat == unknown:
                        # if True:
                        #     performance += 1
                        reward += 5
                        predicted_solvability = self.predictor.predict(self.smtlib_str)
                        if predicted_solvability == 0:
                            performance += 1
                            # 提高一下reward数值
                            reward += 15
                            r = solver.check()
                            stats = solver.statistics()
                            if z3.sat == r:
                                performance += 1
                                reward += 20
                                self.finish = True
                                self.solve_time = stats.get_key_value('time')
                                print("求解时间:", stats.get_key_value('time'))
                            else:
                                # reward += 1 / stats.get_key_value('time') * 100
                                reward += -20
                        else:
                            reward += -15
                    else:
                        # reward += 1 / stats.get_key_value('time') * 100
                        reward += -10
                else:
                    reward += -10
        else:
            # 没有反例的情况下：
            solver_part = Solver()
            assertions = solver.assertions()

            assertions_list = []
            for a in assertions:
                assertions_list.append(a)

            indexes = random.sample(range(len(assertions_list)), int(len(assertions) * 0.5))

            # 根据索引列表，从原始列表中选取元素，并保持原始顺序
            res = [assertions_list[i] for i in sorted(indexes)]
            # res = random.sample(assertions_list, int(len(assertions) * 0.6))
            for r in res:
                solver_part.add(r)
            predicted_solvability_part = self.predictor.predict(solver_part.to_smt2())
            if predicted_solvability_part == 0:
                # if True:
                performance += 1
                reward += 10
                # 注释掉提高速度
                # solver_part.set("timeout", 60000)
                # r = solver_part.check()
                # if z3.sat == r:
                solver_part.set("timeout", 30000)
                r = solver_part.check()
                if z3.sat == r:
                    # if True:
                    performance += 1
                    reward += 10
                    predicted_solvability = self.predictor.predict(self.smtlib_str)
                    if predicted_solvability == 0:
                        performance += 1
                        # 提高一下reward数值
                        reward += 15
                        r = solver.check()
                        stats = solver.statistics()
                        if z3.sat == r:
                            performance += 1
                            reward += 20
                            self.finish = True
                            self.solve_time = stats.get_key_value('time')
                            print("求解时间:", stats.get_key_value('time'))
                        else:
                            # reward += 1 / stats.get_key_value('time') * 100
                            reward += -20
                    else:
                        reward += -15
                elif z3.sat == unknown:
                    # if True:
                    #     performance += 1
                    reward += 5
                    predicted_solvability = self.predictor.predict(self.smtlib_str)
                    if predicted_solvability == 0:
                        performance += 1
                        # 提高一下reward数值
                        reward += 15
                        r = solver.check()
                        stats = solver.statistics()
                        if z3.sat == r:
                            performance += 1
                            reward += 20
                            self.finish = True
                            self.solve_time = stats.get_key_value('time')
                            print("求解时间:", stats.get_key_value('time'))
                        else:
                            # reward += 1 / stats.get_key_value('time') * 100
                            reward += -20
                    else:
                        reward += -15
                else:
                    # reward += 1 / stats.get_key_value('time') * 100
                    reward += -10
            else:
                reward += -10
        # query_smt2 = solver.to_smt2()
        # print(query_smt2)
        if performance < self.last_performance:
            self.reset()
        self.last_performance = performance
        return reward

    def are_lists_equal(self, list1, list2):
        if len(list1) != len(list2):
            return False

        for item1, item2 in zip(list1, list2):
            if item1 != item2:
                return False

        return True

    def render(self) -> None:
        """Renders the environment. Default implementation does nothing."""
        return None

    def close(self) -> None:
        """
        Closes environment, taking care of any cleanup needed.
        Default implementation does nothing.
        """
        return None


class ConstraintSimplificationEnv_v3(Environment):

    def __init__(self, embedder, z3ast, num_variables, num_constants, smtlib_str):
        self.actions_v = None
        self.embedder = embedder
        self.z3ast = z3ast
        self.z3ast_original = z3ast
        self.num_variables = num_variables
        self.num_constants = num_constants
        self.smtlib_str = smtlib_str
        self.smtlib_str_original = smtlib_str
        self.state_original = self.embedder.get_max_pooling_embedding(self.smtlib_str)
        self.state = None
        self.variables = set()
        self.actions = []
        self.concrete_finish = False
        self.concrete_count = 0
        self.counterexamples_list = []
        self.finish = False
        self.used_variables = []
        # 记录state输入了多少次
        self.state_count = 0
        self.predictor = Predictor('KNN')
        self.last_performance = 0

    def reset(self, seed=None):
        self.concrete_finish = False
        self.concrete_count = 0
        # self.finish = False
        self.used_variables = []
        # 从原始的ast开始构建s
        self.state = self.state_original
        # self.state = self.embedder.get_max_pooling_embedding(self.smtlib_str)
        self.z3ast = self.z3ast_original
        self.smtlib_str = self.smtlib_str_original
        self.last_performance = 0
        # graph = embedding_util.Z3ASTGraph(self.z3ast_original)
        # # node_type_dict = NODE_TYPE_ENUM
        # graph2vec = embedding_util.Graph2Vec(graph)
        # # 步骤5: 输出转换结果
        # # node_embed = embedding_util.glorot_uniform(graph2vec.node_feat)
        # node_embed = Parameter(graph2vec.node_feat)
        # self.state = self.encoder(node_embed)

        # variables = set()
        # for a in self.z3ast:
        #     visit(a, variables)
        # self.variables = list(variables)

        self.variables = extract_variables_from_smt2_content(self.smtlib_str)
        # 之后要修改成变量+常量
        # for i in range(len(self.variables)):
        #     self.actions.append(torch.tensor(i))
        #     # 先不使用字典了
        # tensor = torch.arange(-10000, 10001)
        # 笛卡尔积
        self.actions_v = self.strings_to_onehot(self.variables)

        self.actions = get_actions(self.actions_v, torch.arange(0, len(dict_value) - 1))

        self.actions.to(device)
        # self.variables = {index: item for index, item in enumerate(self.variables)}
        # self.action_space = BoxActionSpace([torch.tensor(0), torch.tensor(-10000)],
        #                                    [torch.tensor(len(self.variables)), torch.tensor(10000)])
        self.action_space = DiscreteActionSpace(self.actions)
        del self.actions
        torch.cuda.empty_cache()
        return self.state, self.action_space

    def action_space(self):
        """Returns the action space of the environment."""
        pass

    def step(self, action):
        # print(action)
        try:
            reward = 0
            # variable_pred = self.variables[action]
            # action = self.action_space.
            action = self.action_space.actions_batch[action]
            action_v = action[:-1]
            action_n = action[-1]
            # print(self.onehot_to_indices(action_v))
            variable_pred = self.variables[self.onehot_to_indices(action_v)]
            # print(self.counterexamples_list)
            print(variable_pred)
            # 在一次执行过程中，action不能重复
            if self.concrete_count == 0:
                self.counterexamples_list.append([])
            if variable_pred not in self.used_variables:
                self.used_variables.append(variable_pred)
                self.concrete_count += 1
                # 数值这部分需要修改
                # print(action_n.item)
                # print(type(action_n.item))
                selected_int = int(dict_value[str(int(action_n.item()))])
                print(selected_int)
                self.counterexamples_list[-1].append([variable_pred, selected_int])

                smtlib_str_before, smtlib_str_after = split_at_check_sat(self.smtlib_str)

                type_info = find_var_declaration_in_string(self.smtlib_str_original, variable_pred)
                print(type_info)
                print(type(type_info))
                type_scale = type_info.split(' ')[-1]
                print(type_scale)
                # if type_info == '_ BitVec 64':
                #     new_constraint = "(assert (= {} (_ bv{} 64)))\n".format(variable_pred, selected_int)
                # elif type_info == '_ BitVec 8':
                #     new_constraint = "(assert (= {} (_ bv{} 8)))\n".format(variable_pred, selected_int)
                # elif type_info == '_ BitVec 1008':
                #     new_constraint = "(assert (= {} (_ bv{} 1008)))\n".format(variable_pred, selected_int)
                new_constraint = "(assert (= {} (_ bv{} {})))\n".format(variable_pred, selected_int, type_scale)
                self.smtlib_str = smtlib_str_before + new_constraint + smtlib_str_after
                assertions = parse_smt2_string(self.smtlib_str)
                solver = Solver()
                for a in assertions:
                    solver.add(a)

                # solver = Solver()
                # for a in self.z3ast:
                #     solver.add(a)
                # # change name
                # v_name = 'v_name'
                # exec(f"{v_name} = Int('{variable_pred}')")
                # # 修改，添加取值部分内容
                # solver.add(eval(v_name) == selected_int)
                # print(solver)

                reward += self.calculate_reward(solver)
                self.z3ast = solver.assertions()
                # graph = embedding_util.Z3ASTGraph(self.z3ast)
                # # node_type_dict = NODE_TYPE_ENUM
                # graph2vec = embedding_util.Graph2Vec(graph)
                # # 步骤5: 输出转换结果
                # print("节点特征向量:")
                # print(graph2vec.node_feat.shape)
                # # node_embed = embedding_util.glorot_uniform(graph2vec.node_feat)
                # node_embed = Parameter(graph2vec.node_feat)
                # print(solver.to_smt2())
                # print(type(solver.to_smt2()))
                self.state = self.embedder.get_max_pooling_embedding(solver.to_smt2())

                if self.concrete_count == len(self.variables):
                    self.concrete_finish = True
                    self.reset()
                    # 判断这里需不需要直接reset
            else:
                reward += -10
                print(action)
                # 重新实现
                # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                # device = torch.device("cpu")
                self.actions_v = [act.to(device) for act in self.actions_v]
                self.actions_v = [torch.round(act) for act in self.actions_v]
                action_v = [act.to(device) for act in action_v]
                action_v = [torch.round(act) for act in action_v]

                # print(self.actions)
                # for i in self.actions_v:
                #     i.to(device)
                # for i in action_v:
                #     i.to(device)
                # action_v.to(device)
                self.actions_v = [tensor1 for tensor1 in self.actions_v if
                                  not any(torch.equal(tensor1, tensor2) for tensor2 in action_v)]
                self.action_space = DiscreteActionSpace(
                    get_actions(self.actions_v, torch.arange(0, len(dict_value) - 1)))
            # print('***********************')
            # print(len(self.counterexamples_list))
            # for i in self.counterexamples_list:
            #     print(len(i))
            # 清除内存
            del action
            del action_n
            del action_v
            torch.cuda.empty_cache()
        except Exception as e:
            print('some problems are triggered')
            self.state = self.state_original
            reward = 0
        return ActionResult(
            observation=self.state,
            reward=float(reward),
            terminated=self.finish,
            truncated=False,
            info={},
            available_action_space=self.action_space, )

    @staticmethod
    def strings_to_onehot(string_list):
        # 创建一个从字符串到索引的映射
        str_to_index = {string: index for index, string in enumerate(string_list)}

        # 创建One-Hot编码的张量
        one_hot_tensors = []
        for string in string_list:
            # 创建一个全0的向量
            one_hot_vector = torch.zeros(len(string_list), dtype=torch.float32)
            # 将对应位置置1
            one_hot_vector[str_to_index[string]] = 1.0
            one_hot_vector.to(device)
            one_hot_tensors.append(one_hot_vector)
        one_hot_matrix = torch.stack(one_hot_tensors)
        del one_hot_vector
        del one_hot_tensors
        torch.cuda.empty_cache()
        return one_hot_matrix
        # return one_hot_tensors

    @staticmethod
    def onehot_to_indices(one_hot_tensors):
        # 将One-Hot编码的张量转换回索引
        return torch.argmax(one_hot_tensors).item()

    @staticmethod
    def counter_reward_function(total_length, unique_count):

        """
        Calculate the reward based on the total length of the list and the number of unique in it.

        Args:
        - total_length (int): The total length of the list.
        - unique_count (int): The number of unique in the list.

        Returns:
        - float: The calculated reward.
        """
        # Define the base reward values
        R_positive = 1
        R_negative = -1

        # Define the scaling factor for negative reward
        alpha = 1 / math.sqrt(total_length) if total_length > 0 else 1

        # Check if there are any unique strings
        if unique_count > 0:
            # Calculate the positive reward, scaled based on the list length
            reward = R_positive / math.log(1 + total_length)
        else:
            # Apply the negative reward, scaled by alpha
            reward = R_negative * alpha

        return reward

    def calculate_reward(self, solver):
        performance = 0
        reward = 0
        count = 0
        solver.set("timeout", 60000)
        # 判断新产生的序列和之前有没有重复
        # 判断是否存在反例
        if len(self.counterexamples_list) > 1:
            if self.counterexamples_list[-1] in self.counterexamples_list[:len(self.counterexamples_list) - 1]:
                reward += -1
            else:
                # 判断新的序列和之前是否有重复（字符串重复）
                # for i in range(len(self.counterexamples_list) - 1):
                #     # if self.are_lists_equal(self.counterexamples_list[i],self.counterexamples_list[-1]):
                #     if ' '.join(self.counterexamples_list[-1]) in ' '.join(self.counterexamples_list[i]):
                #         count += 1
                last_joined = ' '.join(
                    ' '.join(str(item) for item in inner_list) for inner_list in self.counterexamples_list[-1])
                for i in range(len(self.counterexamples_list) - 1):
                    current_joined = ' '.join(
                        ' '.join(str(item) for item in inner_list) for inner_list in self.counterexamples_list[i])
                    if last_joined in current_joined:
                        count += 1
                reward += self.counter_reward_function(len(self.counterexamples_list) - 1,
                                                       len(self.counterexamples_list) - 1 - count)
            print(self.counterexamples_list)
        # 后续实现一些子集求解
        solver_part = Solver()
        assertions = solver.assertions()
        assertions_list = []
        for a in assertions:
            assertions_list.append(a)
        res = random.sample(assertions_list, int(len(assertions) * 0.6))
        for r in res:
            solver_part.add(r)
        predicted_solvability_part = self.predictor.predict(solver_part.to_smt2())
        if predicted_solvability_part == 0:
            performance += 1
            reward += 2
            r = solver_part.check()
            if z3.sat == r:
                performance += 1
                reward += 5
                predicted_solvability = self.predictor.predict(self.smtlib_str)
                if predicted_solvability == 0:
                    performance += 1
                    # 提高一下reward数值
                    reward += 7
                    r = solver.check()
                    stats = solver.statistics()
                    if z3.sat == r:
                        performance += 1
                        reward += 15
                        self.finish = True

                        print("求解时间:", stats.get_key_value('time'))
                        update_txt_with_current_time('time.txt', stats.get_key_value('time'))
                    else:
                        # reward += 1 / stats.get_key_value('time') * 100
                        reward += -15
                else:
                    reward += -7
            else:
                # reward += 1 / stats.get_key_value('time') * 100
                reward += -5
        else:
            reward += -2
        # query_smt2 = solver.to_smt2()
        # print(query_smt2)
        if performance < self.last_performance:
            self.reset()
        self.last_performance = performance
        return reward

    def are_lists_equal(self, list1, list2):
        if len(list1) != len(list2):
            return False

        for item1, item2 in zip(list1, list2):
            if item1 != item2:
                return False

        return True

    def render(self) -> None:
        """Renders the environment. Default implementation does nothing."""
        return None

    def close(self) -> None:
        """
        Closes environment, taking care of any cleanup needed.
        Default implementation does nothing.
        """
        return None


class ConstraintSimplificationEnv_v2(Environment):

    def __init__(self, encoder, decoder, z3ast, num_variables, num_constants):
        self.actions_v = None
        self.decoder = decoder
        self.encoder = encoder
        self.z3ast = z3ast
        self.z3ast_original = z3ast
        self.num_variables = num_variables
        self.num_constants = num_constants
        self.state = None
        self.variables = set()
        self.actions = []
        self.concrete_finish = False
        self.concrete_count = 0
        self.counterexamples_list = []
        self.finish = False
        self.used_variables = []
        # 记录state输入了多少次
        self.state_count = 0
        self.predictor = Predictor('KNN')

    def reset(self, seed=None):
        self.concrete_finish = False
        self.concrete_count = 0
        self.finish = False
        self.used_variables = []
        # 从原始的ast开始构建s

        graph = embedding_util.Z3ASTGraph(self.z3ast_original)
        # node_type_dict = NODE_TYPE_ENUM
        graph2vec = embedding_util.Graph2Vec(graph)
        # 步骤5: 输出转换结果
        # node_embed = embedding_util.glorot_uniform(graph2vec.node_feat)
        node_embed = Parameter(graph2vec.node_feat)
        self.state = self.encoder(node_embed)
        variables = set()
        for a in self.z3ast:
            visit(a, variables)
        self.variables = list(variables)
        # 之后要修改成变量+常量
        # for i in range(len(self.variables)):
        #     self.actions.append(torch.tensor(i))
        #     # 先不使用字典了
        # tensor = torch.arange(-10000, 10001)
        # 笛卡尔积
        self.actions_v = self.strings_to_onehot(self.variables)
        self.actions = get_actions(self.actions_v, torch.arange(-10000, 10001))

        self.actions.to(device)
        # self.variables = {index: item for index, item in enumerate(self.variables)}
        # self.action_space = BoxActionSpace([torch.tensor(0), torch.tensor(-10000)],
        #                                    [torch.tensor(len(self.variables)), torch.tensor(10000)])
        self.action_space = DiscreteActionSpace(self.actions)
        return self.state, self.action_space

    def action_space(self):
        """Returns the action space of the environment."""
        pass

    def step(self, action):
        reward = 0
        # variable_pred = self.variables[action]
        # action = self.action_space.
        action = self.action_space.actions_batch[action]
        action_v = action[:-1]
        action_n = action[-1]
        variable_pred = self.variables[self.onehot_to_indices(action_v)[0]]
        # 在一次执行过程中，action不能重复
        if self.concrete_count == 0:
            self.counterexamples_list.append([])
        if variable_pred not in self.used_variables:
            self.used_variables.append(variable_pred)
            self.concrete_count += 1
            # 数值这部分需要修改
            # min_int32 = -2147483648
            # max_int32 = 2147483647
            #
            # # 生成一个随机的32位整数
            # random_int = random.randint(min_int32, max_int32)
            selected_int = action_n.item()
            self.counterexamples_list[-1].append([variable_pred, selected_int])

            solver = Solver()
            for a in self.z3ast:
                solver.add(a)
            exec(f"{variable_pred} = Int('{variable_pred}')")
            # 修改，添加取值部分内容

            solver.add(eval(variable_pred) == selected_int)

            reward += self.calculate_reward(solver)
            self.z3ast = solver.assertions()
            graph = embedding_util.Z3ASTGraph(self.z3ast)
            # node_type_dict = NODE_TYPE_ENUM
            graph2vec = embedding_util.Graph2Vec(graph)
            # 步骤5: 输出转换结果
            print("节点特征向量:")
            print(graph2vec.node_feat.shape)
            # node_embed = embedding_util.glorot_uniform(graph2vec.node_feat)
            node_embed = Parameter(graph2vec.node_feat)
            self.state = self.encoder(node_embed)

            if self.concrete_count == len(self.variables):
                self.concrete_finish = True
                self.reset()
                # 判断这里需不需要直接reset
        else:
            reward += -10
            print(action)
            # 重新实现
            # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            device = torch.device("cpu")
            self.actions_v = [act.to(device) for act in self.actions_v]
            action_v = [act.to(device) for act in action_v]
            print(self.actions)
            for i in self.actions_v:
                i.to(device)
            for i in action_v:
                i.to(device)
            # action_v.to(device)
            self.actions_v = [tensor1 for tensor1 in self.actions_v if
                              not any(torch.equal(tensor1, tensor2) for tensor2 in action_v)]
            self.action_space = DiscreteActionSpace(get_actions(self.actions_v, torch.arange(-10000, 10001)))
        # 清除内存
        torch.cuda.empty_cache()
        return ActionResult(
            observation=self.state,
            reward=float(reward),
            terminated=self.concrete_finish,
            truncated=self.finish,
            info={},
            available_action_space=self.action_space, )

    @staticmethod
    def strings_to_onehot(string_list):
        # 创建一个从字符串到索引的映射
        str_to_index = {string: index for index, string in enumerate(string_list)}

        # 创建One-Hot编码的张量
        one_hot_tensors = []
        for string in string_list:
            # 创建一个全0的向量
            one_hot_vector = torch.zeros(len(string_list), dtype=torch.float32)
            # 将对应位置置1
            one_hot_vector[str_to_index[string]] = 1.0
            one_hot_vector.to(device)
            one_hot_tensors.append(one_hot_vector)
        one_hot_matrix = torch.stack(one_hot_tensors)

        return one_hot_matrix
        # return one_hot_tensors

    @staticmethod
    def onehot_to_indices(one_hot_tensors):
        # 将One-Hot编码的张量转换回索引
        return [torch.argmax(tensor).item() for tensor in one_hot_tensors]

    @staticmethod
    def counter_reward_function(total_length, unique_count):

        """
        Calculate the reward based on the total length of the list and the number of unique in it.

        Args:
        - total_length (int): The total length of the list.
        - unique_count (int): The number of unique in the list.

        Returns:
        - float: The calculated reward.
        """
        # Define the base reward values
        R_positive = 1
        R_negative = -1

        # Define the scaling factor for negative reward
        alpha = 1 / math.sqrt(total_length) if total_length > 0 else 1

        # Check if there are any unique strings
        if unique_count > 0:
            # Calculate the positive reward, scaled based on the list length
            reward = R_positive / math.log(1 + total_length)
        else:
            # Apply the negative reward, scaled by alpha
            reward = R_negative * alpha

        return reward

    def calculate_reward(self, solver):
        reward = 0
        count = 0
        solver.set("timeout", 60000)
        # 判断新产生的序列和之前有没有重复
        # 判断是否存在反例
        if len(self.counterexamples_list) > 1:
            if self.counterexamples_list[-1] in self.counterexamples_list[:len(self.counterexamples_list) - 1]:
                reward += -1
            else:
                # 判断新的序列和之前是否有重复（字符串重复）
                # for i in range(len(self.counterexamples_list) - 1):
                #     # if self.are_lists_equal(self.counterexamples_list[i],self.counterexamples_list[-1]):
                #     if ' '.join(self.counterexamples_list[-1]) in ' '.join(self.counterexamples_list[i]):
                #         count += 1
                last_joined = ' '.join(
                    ' '.join(str(item) for item in inner_list) for inner_list in self.counterexamples_list[-1])
                for i in range(len(self.counterexamples_list) - 1):
                    current_joined = ' '.join(
                        ' '.join(str(item) for item in inner_list) for inner_list in self.counterexamples_list[i])
                    if last_joined in current_joined:
                        count += 1
                reward += self.counter_reward_function(len(self.counterexamples_list) - 1,
                                                       len(self.counterexamples_list) - 1 - count)
            print(self.counterexamples_list)
        # 后续实现一些子集求解
        query_smt2 = solver.to_smt2()
        # print(query_smt2)
        predicted_solvability = self.predictor.predict(query_smt2)
        if predicted_solvability == 0:
            # 提高一下reward数值
            reward += 2
            r = solver.check()
            stats = solver.statistics()
            if z3.sat == r:

                self.finish = True

                print("求解时间:", stats.get_key_value('time'))
            else:
                # reward += 1 / stats.get_key_value('time') * 100
                reward += -5

        return reward

    def are_lists_equal(self, list1, list2):
        if len(list1) != len(list2):
            return False

        for item1, item2 in zip(list1, list2):
            if item1 != item2:
                return False

        return True

    def render(self) -> None:
        """Renders the environment. Default implementation does nothing."""
        return None

    def close(self) -> None:
        """
        Closes environment, taking care of any cleanup needed.
        Default implementation does nothing.
        """
        return None


class ConstraintSimplificationEnv_v4(Environment):

    def __init__(self, embedder, z3ast, num_variables, num_constants, smtlib_str):
        self.actions_v = None
        self.embedder = embedder
        self.z3ast = z3ast
        self.z3ast_original = z3ast
        self.num_variables = num_variables
        self.num_constants = num_constants
        self.smtlib_str = smtlib_str
        self.state = None
        self.variables = set()
        self.actions = []
        self.concrete_finish = False
        self.concrete_count = 0
        self.counterexamples_list = []
        self.finish = False
        self.used_variables = []
        # 记录state输入了多少次
        self.state_count = 0
        self.predictor = Predictor('KNN')

    def reset(self, seed=None):
        self.concrete_finish = False
        self.concrete_count = 0
        self.finish = False
        self.used_variables = []
        # 从原始的ast开始构建s
        self.state = self.embedder.get_max_pooling_embedding(self.smtlib_str)
        self.z3ast = self.z3ast_original
        # graph = embedding_util.Z3ASTGraph(self.z3ast_original)
        # # node_type_dict = NODE_TYPE_ENUM
        # graph2vec = embedding_util.Graph2Vec(graph)
        # # 步骤5: 输出转换结果
        # # node_embed = embedding_util.glorot_uniform(graph2vec.node_feat)
        # node_embed = Parameter(graph2vec.node_feat)
        # self.state = self.encoder(node_embed)

        # variables = set()
        # for a in self.z3ast:
        #     visit(a, variables)
        # self.variables = list(variables)

        self.variables = extract_variables_from_smt2_content(self.smtlib_str)
        # 之后要修改成变量+常量
        # for i in range(len(self.variables)):
        #     self.actions.append(torch.tensor(i))
        #     # 先不使用字典了
        # tensor = torch.arange(-10000, 10001)
        # 笛卡尔积
        self.actions_v = self.strings_to_onehot(self.variables)

        self.actions = get_actions(self.actions_v, torch.arange(0, len(dict_value) - 1))

        self.actions.to(device)
        # self.variables = {index: item for index, item in enumerate(self.variables)}
        # self.action_space = BoxActionSpace([torch.tensor(0), torch.tensor(-10000)],
        #                                    [torch.tensor(len(self.variables)), torch.tensor(10000)])
        self.action_space = DiscreteActionSpace(self.actions)
        del self.actions
        torch.cuda.empty_cache()
        return self.state, self.action_space

    def action_space(self):
        """Returns the action space of the environment."""
        pass

    def step(self, action):
        reward = 0
        # variable_pred = self.variables[action]
        # action = self.action_space.
        action = self.action_space.actions_batch[action]
        action_v = action[:-1]
        action_n = action[-1]
        variable_pred = self.variables[self.onehot_to_indices(action_v)[0]]
        # 在一次执行过程中，action不能重复
        if self.concrete_count == 0:
            self.counterexamples_list.append([])
        if variable_pred not in self.used_variables:
            self.used_variables.append(variable_pred)
            self.concrete_count += 1
            # 数值这部分需要修改
            # print(action_n.item)
            # print(type(action_n.item))
            print()
            # selected_int = int(dict_value[str(int(action_n.item()))])
            selected_int = random.choice(dict_value[int(action_n.item())])
            self.counterexamples_list[-1].append([variable_pred, selected_int])

            solver = Solver()
            for a in self.z3ast:
                solver.add(a)
            # change name
            v_name = 'v_name'
            exec(f"{v_name} = Int('{variable_pred}')")
            # 修改，添加取值部分内容

            solver.add(eval(v_name) == selected_int)
            print(solver)
            reward += self.calculate_reward(solver)
            self.z3ast = solver.assertions()
            # graph = embedding_util.Z3ASTGraph(self.z3ast)
            # # node_type_dict = NODE_TYPE_ENUM
            # graph2vec = embedding_util.Graph2Vec(graph)
            # # 步骤5: 输出转换结果
            # print("节点特征向量:")
            # print(graph2vec.node_feat.shape)
            # # node_embed = embedding_util.glorot_uniform(graph2vec.node_feat)
            # node_embed = Parameter(graph2vec.node_feat)
            self.state = self.embedder.get_max_pooling_embedding(solver.to_smt2())

            if self.concrete_count == len(self.variables):
                self.concrete_finish = True
                self.reset()
                # 判断这里需不需要直接reset
        else:
            reward += -10
            print(action)
            # 重新实现
            # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            device = torch.device("cpu")
            self.actions_v = [act.to(device) for act in self.actions_v]
            action_v = [act.to(device) for act in action_v]
            # print(self.actions)
            for i in self.actions_v:
                i.to(device)
            for i in action_v:
                i.to(device)
            # action_v.to(device)
            self.actions_v = [tensor1 for tensor1 in self.actions_v if
                              not any(torch.equal(tensor1, tensor2) for tensor2 in action_v)]
            self.action_space = DiscreteActionSpace(get_actions(self.actions_v, torch.arange(0, len(dict_value) - 1)))
        # 清除内存
        del action
        del action_n
        del action_v
        torch.cuda.empty_cache()
        return ActionResult(
            observation=self.state,
            reward=float(reward),
            terminated=self.finish,
            truncated=False,
            info={},
            available_action_space=self.action_space, )

    @staticmethod
    def strings_to_onehot(string_list):
        # 创建一个从字符串到索引的映射
        str_to_index = {string: index for index, string in enumerate(string_list)}

        # 创建One-Hot编码的张量
        one_hot_tensors = []
        for string in string_list:
            # 创建一个全0的向量
            one_hot_vector = torch.zeros(len(string_list), dtype=torch.float32)
            # 将对应位置置1
            one_hot_vector[str_to_index[string]] = 1.0
            one_hot_vector.to(device)
            one_hot_tensors.append(one_hot_vector)
        one_hot_matrix = torch.stack(one_hot_tensors)
        del one_hot_vector
        del one_hot_tensors
        torch.cuda.empty_cache()
        return one_hot_matrix
        # return one_hot_tensors

    @staticmethod
    def onehot_to_indices(one_hot_tensors):
        # 将One-Hot编码的张量转换回索引
        return [torch.argmax(tensor).item() for tensor in one_hot_tensors]

    @staticmethod
    def counter_reward_function(total_length, unique_count):

        """
        Calculate the reward based on the total length of the list and the number of unique in it.

        Args:
        - total_length (int): The total length of the list.
        - unique_count (int): The number of unique in the list.

        Returns:
        - float: The calculated reward.
        """
        # Define the base reward values
        R_positive = 1
        R_negative = -1

        # Define the scaling factor for negative reward
        alpha = 1 / math.sqrt(total_length) if total_length > 0 else 1

        # Check if there are any unique strings
        if unique_count > 0:
            # Calculate the positive reward, scaled based on the list length
            reward = R_positive / math.log(1 + total_length)
        else:
            # Apply the negative reward, scaled by alpha
            reward = R_negative * alpha

        return reward

    def calculate_reward(self, solver):
        reward = 0
        count = 0
        solver.set("timeout", 60000)
        # 判断新产生的序列和之前有没有重复
        # 判断是否存在反例
        if len(self.counterexamples_list) > 1:
            if self.counterexamples_list[-1] in self.counterexamples_list[:len(self.counterexamples_list) - 1]:
                reward += -1
            else:
                # 判断新的序列和之前是否有重复（字符串重复）
                # for i in range(len(self.counterexamples_list) - 1):
                #     # if self.are_lists_equal(self.counterexamples_list[i],self.counterexamples_list[-1]):
                #     if ' '.join(self.counterexamples_list[-1]) in ' '.join(self.counterexamples_list[i]):
                #         count += 1
                last_joined = ' '.join(
                    ' '.join(str(item) for item in inner_list) for inner_list in self.counterexamples_list[-1])
                for i in range(len(self.counterexamples_list) - 1):
                    current_joined = ' '.join(
                        ' '.join(str(item) for item in inner_list) for inner_list in self.counterexamples_list[i])
                    if last_joined in current_joined:
                        count += 1
                reward += self.counter_reward_function(len(self.counterexamples_list) - 1,
                                                       len(self.counterexamples_list) - 1 - count)
            print(self.counterexamples_list)
        # 后续实现一些子集求解
        query_smt2 = solver.to_smt2()
        # print(query_smt2)
        predicted_solvability = self.predictor.predict(query_smt2)
        if predicted_solvability == 0:
            # 提高一下reward数值
            reward += 2
            r = solver.check()
            stats = solver.statistics()
            if z3.sat == r:

                self.finish = True

                print("求解时间:", stats.get_key_value('time'))
                update_txt_with_current_time('time.txt', self.smtlib_str, stats.get_key_value('time'))
            else:
                # reward += 1 / stats.get_key_value('time') * 100
                reward += -5

        return reward

    def are_lists_equal(self, list1, list2):
        if len(list1) != len(list2):
            return False

        for item1, item2 in zip(list1, list2):
            if item1 != item2:
                return False

        return True

    def render(self) -> None:
        """Renders the environment. Default implementation does nothing."""
        return None

    def close(self) -> None:
        """
        Closes environment, taking care of any cleanup needed.
        Default implementation does nothing.
        """
        return None


def visit(expr, variables):
    if is_const(expr) and expr.decl().kind() == Z3_OP_UNINTERPRETED:
        # Add only uninterpreted functions (which represent variables)
        # print(type(self.variables))
        variables.add(str(expr))
    else:
        # Recursively visit children for composite expressions
        for child in expr.children():
            visit(child, variables)


def get_actions(tensor_2d, tensor_1d):
    import itertools
    # 示例二维张量和一维张量
    # tensor_2d = torch.tensor([[1, 2], [3, 4]])
    # tensor_1d = torch.tensor([5, 6])
    # 将二维张量的每行转换为元组，并与一维张量的每个元素结合
    cartesian_product = list(itertools.product(tensor_2d, tensor_1d))
    # 将结果转换回张量
    result = torch.tensor([[*tup[0].clone().cpu().numpy(), tup[1].clone().cpu().item()] for tup in cartesian_product])
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    result.to(device)
    del cartesian_product
    torch.cuda.empty_cache()
    print('*********************')
    print(result, type(result))
    return result


import re


def extract_variables_from_smt2_content(content):
    """
    从 SMT2 格式的字符串内容中提取变量名，排除布尔类型的变量。

    参数:
    - content: SMT2 格式的字符串内容。

    返回:
    - 非布尔类型变量名列表。
    """
    # 用于匹配 `(declare-fun ...)` 语句的正则表达式，包括变量名和类型
    variable_pattern = re.compile(r'\(declare-fun\s+([^ ]+)\s*\(\s*\)\s*([^)]+)\)')

    # 存储提取的非布尔类型变量名
    variables = []

    # 按行分割字符串并迭代每一行
    for line in content.splitlines():
        # 在每一行中查找匹配的变量声明
        match = variable_pattern.search(line)
        if match:
            var_name, var_type = match.group(1, 2)
            # 如果变量类型不是 Bool，则将变量名添加到列表中
            if var_type != 'Bool':
                variables.append(var_name.replace('|', ''))

    return variables
