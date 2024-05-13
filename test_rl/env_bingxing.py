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
    find_assertions_related_to_var_names_optimized_dfs, save_string_to_file

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0")

import torch
import numpy as np
import random
from pearl.api.action_result import ActionResult
from pearl.api.environment import Environment
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace
import datetime

import multiprocessing
import threading
import time
import os


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



class ConstraintSimplificationEnv_test(Environment):

    def __init__(self, embedder, z3ast, num_variables, num_constants, smtlib_str, file_path):

        self.solve_v = None
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
        self.agree_examples = []
        self.finish = False
        self.used_variables = []
        # 记录state输入了多少次
        self.state_count = 0
        self.predictor = Predictor('KNN')
        self.last_performance = 0
        self.solve_time = 0
        self.v_related_assertions = find_assertions_related_to_var_names_optimized_dfs(self.z3ast, self.variables)

        self.process = None
        self.threads = []
        self.stop_event = multiprocessing.Event()
        self.cpu_cores = os.cpu_count()  # 获取CPU核心数
        self.active_threads = 0  # 当前活跃的线程数
        self.condition = multiprocessing.Condition()
        self.process_flag = False
        # self.start()

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
                #找到最大最小值
                if min_value <= selected_int <= max_value:
                    reward += 5
                    #需要添加的约束
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

                        save_string_to_file('smt.json', new_smtlib_str)
                        if predicted_solvability_related == 0:

                            reward += 5
                            assertions_related = parse_smt2_string(new_smtlib_str)
                            solver_related = Solver()
                            for a in assertions_related:
                                solver_related.add(a)
                            solver_related.set("timeout", 10000)
                            r = solver_related.check()
                            if z3.sat == r:

                                reward += 5
                                self.used_variables.append(variable_pred)
                                self.concrete_count += 1
                                # 数值这部分需要修改
                                # print(action_n.item)
                                # print(type(action_n.item)
                            elif z3.unknown == r:
                                reward += 2
                                self.used_variables.append(variable_pred)
                                self.concrete_count += 1
                                # 数值这部分需要修改
                                # print(action_n.item)
                                # print(type(action_n.item))


                            else:
                                reward += -5
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
                    else:#没有变量约束的情况
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
        if self.step_count > 1000:
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
                save_string_to_file('smt.json', solver_part.to_smt2())
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
                        save_string_to_file('smt.json', self.smtlib_str)
                        if predicted_solvability == 0:
                            #需要进行验证的部分
                            self.agree_examples.append([])
                            self.agree_examples[-1] = {}
                            self.agree_examples[-1]['v'] = self.counterexamples_list[-1]
                            self.agree_examples[-1]['smtstr'] = self.smtlib_str
                            self.agree_examples[-1]['reward'] = reward
                            performance += 1
                            # 提高一下reward数值
                            reward += 15
                            if len(self.agree_examples) > 0:
                                self.start_thread()
                            # if self.process_flag:
                            #     pass
                            # else:
                            #     self.start()
                            #     self.process_flag = True

                            # r = solver.check()
                            # stats = solver.statistics()
                            # if z3.sat == r:
                            #     performance += 1
                            #     reward += 20
                            #     self.finish = True
                            #     self.solve_time = stats.get_key_value('time')
                            #     print("求解时间:", stats.get_key_value('time'))
                            # else:
                            #     # reward += 1 / stats.get_key_value('time') * 100
                            #     reward += -20
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

                # solver_part.set("timeout", 30000)
                # r = solver_part.check()
                # if z3.sat == r:
                if True:
                    # performance += 1
                    # reward += 10
                    predicted_solvability = self.predictor.predict(self.smtlib_str)
                    if predicted_solvability == 0:

                        # 需要进行验证的部分
                        self.agree_examples.append([])
                        self.agree_examples[-1] = {}
                        self.agree_examples[-1]['v'] = self.counterexamples_list[-1]
                        self.agree_examples[-1]['smtstr'] = self.smtlib_str
                        self.agree_examples[-1]['reward'] = reward

                        if len(self.agree_examples) > 0:
                            self.start_thread()

                        # if self.process_flag:
                        #     pass
                        # else:
                        #     self.start()
                        #     self.process_flag = True
                        performance += 1
                        # 提高一下reward数值
                        reward += 15
                        # r = solver.check()
                        # stats = solver.statistics()
                        # if z3.sat == r:
                        #     performance += 1
                        #     reward += 20
                        #     self.finish = True
                        #     self.solve_time = stats.get_key_value('time')
                        #     print("求解时间:", stats.get_key_value('time'))
                        # else:
                        #     # reward += 1 / stats.get_key_value('time') * 100
                        #     reward += -20
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

    def infinite_loop(self):
        with self.condition:
            while not self.stop_event.is_set():
                print('循环进行中。。。')
                print(self.agree_examples)
                if len(self.agree_examples) > 0:
                    print('求解进行中。。。')
                    # 根据活跃线程的数量和CPU核心数决定是否启动新线程
                    if self.active_threads < self.cpu_cores:
                        self.start_thread()
                        self.active_threads += 1
                    elif self.active_threads > 0:
                        # 如果有完成的线程，减少活跃线程数
                        self.check_for_completed_threads()
                    time.sleep(0.5)  # 暂停一段时间，减少CPU使用
                else:
                    print('没有求解的例子')
                    time.sleep(10)
    def start_thread(self):
        # 创建并启动新线程
        def thread_target():
            print(f"Thread {threading.current_thread().name} started.")
            try:
                self.thread_function()
            finally:
                with self.condition:
                    self.active_threads -= 1  # 减少活跃线程数
                    self.condition.notify()  # 通知可能有新线程可以启动
            print(f"Thread {threading.current_thread().name} finished.")

        thread = threading.Thread(target=thread_target)
        thread.start()
        self.threads.append(thread)

    def thread_function(self):
        # 模拟耗时操作
        # print('求解中。。。')
        if len(self.agree_examples) > 0:
            print('开始求解')
            self.agree_examples.sort(key=lambda x: x['reward'], reverse=True)
            assertions = parse_smt2_string(self.agree_examples[0]['smtstr'])
            temp_list = self.agree_examples[0]['v']
            with threading.Lock():
                self.agree_examples.pop(0)
            solver = Solver()
            for a in assertions:
                solver.add(a)
            r = solver.check()
            stats = solver.statistics()
            if z3.sat == r:
                print('求解成功')
                self.finish = True
                self.solve_time = stats.get_key_value('time')
                self.solve_v = temp_list
                # self.stop()
        else:
            print('没有求解的例子')
            time.sleep(5)
        time.sleep(2)

    def check_for_completed_threads(self):
        # 检查并移除已完成的线程
        self.threads = [thread for thread in self.threads if thread.is_alive()]

    def start(self):
        self.process = multiprocessing.Process(target=self.infinite_loop)
        self.process.start()

    def stop(self):
        # 停止无限循环的进程
        self.stop_event.set()
        if self.process is not None:
            # if self.process.is_alive():
            #     self.process.terminate()
            self.process.join()
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

