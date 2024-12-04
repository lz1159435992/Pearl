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
import sys
import traceback
from decimal import Decimal, getcontext
import torch
import z3
from openai import OpenAI
from torch.nn.parameter import Parameter
from z3 import *
import embedding_util
from pearl.SMTimer.KNN_Predictor import Predictor
from pearl.api import Space
from test_rl.test_script.db_search_lz_alue import fetch_data_as_dict
from test_rl.test_script.utils import find_var_declaration_in_string, split_at_check_sat, load_dictionary, \
    find_assertions_related_to_var_name, find_assertions_related_to_var_names_optimized, repalce_veriable, \
    normalize_smt_str_without_replace, \
    solve_assertion_get_range

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0")

import torch
import numpy as np
import random
from pearl.api.action_result import ActionResult
from pearl.api.environment import Environment
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace
import datetime
from bert_predictor_mask import SimpleClassifier
from bert_predictor_2_mask import EnhancedEightClassModel

sys.path.append('/home/nju/PycharmProjects/Pearl/test_rl')
def is_number(s):
    # 匹配整数、小数和分数
    pattern = r'^(\d+|\d+\.\d+|\d+\/\d+)$'
    return re.match(pattern, s) is not None
class ConstraintSimplificationEnv_test(Environment):

    def __init__(self, embedder, z3ast, model, model_time, smtlib_str, file_path, var_dict, constant_list):
        self.range_count = 10000
        self.var_dict = var_dict
        self.constant_list = constant_list
        self.step_count = 0
        self.file_path = file_path
        self.actions_v = None
        self.embedder = embedder
        self.z3ast = z3ast
        self.z3ast_original = copy.deepcopy(z3ast)
        self.smtlib_str = smtlib_str
        self.smtlib_str_original = copy.deepcopy(smtlib_str)
        self.variables = normalize_smt_str_without_replace(self.smtlib_str)
        self.state_original = self.embedder.get_max_pooling_embedding(self.smtlib_str, self.variables)
        self.state = None

        self.actions = []
        self.concrete_finish = False
        self.concrete_count = 0
        self.counterexamples_list = []
        self.finish = False
        self.used_variables = []
        # 记录state输入了多少次
        self.state_count = 0
        self.predictor = model
        self.predictor_time = model_time
        self.last_performance = 0
        self.solve_time = 0
        # self.v_related_assertions,self.var_range_dict = find_assertions_related_to_var_names_optimized_dfs(self.z3ast, self.variables)
        self.v_related_assertions, self.var_range_dict = solve_assertion_get_range(self.z3ast, self.variables)
        #初始化存储范围,在上面初始化了
        # self.var_range_dict = {}
        self.var_range_dict_n = {}
        self.range_init()
        # 直接使用字典字面量来初始化
        self.time_dict = {
            0: 20,  #对于无法求解的约束，简单设置一个时间进行尝试
            1: 1,
            2: 20,
            3: 50,
            4: 100,
            5: 200,
            6: 500,
            7: 1000,

        }
        self.last_action = None

    def range_init(self):
        for variable in self.variables:
            print(self.file_path)
            print(variable)
            type_info = find_var_declaration_in_string(self.smtlib_str_original, variable)
            print(self.smtlib_str_original)
            print(type_info)
            print(type(type_info))
            if 'BitVec' in type_info:
            # if type_info in ['BV']:
                type_scale = type_info.split(' ')[-1]
                print(type_scale)
                max_value = 2 ** int(type_scale) - 1
                min_value = 0
            elif type_info in ['Int', 'Real']:
                # max_value = 2247483648
                # min_value = -2247483648
                max_value = 2147483650
                min_value = -2147483650
                type_scale = 0
            self.var_range_dict[variable].append([min_value, max_value])
            # #添加新的约束
            # for k, v in self.var_bound[variable].items():
            #     for l in v['lower']:
            #         for u in v['upper']:
            #             if l <= u:
            #                 self.var_range_dict[variable].append([l, u])
            # self.var_range_dict_n[variable] = self.divide_intervals_by_n(self.var_range_dict[variable],self.range_count)

    def reset(self, seed=None):
        self.last_action = None
        self.concrete_finish = False
        self.concrete_count = 0
        # self.finish = False
        self.used_variables = []
        # 从原始的ast开始构建s
        self.state = self.state_original.clone().detach()
        # self.state = self.embedder.get_max_pooling_embedding(self.smtlib_str)
        self.z3ast = copy.deepcopy(self.z3ast_original)
        self.smtlib_str = copy.deepcopy(self.smtlib_str_original)
        self.last_performance = 0
        # # self.actions_v = self.strings_to_onehot(self.variables)
        # max_length = 0
        # for k, v in self.var_range_dict_n.items():
        #     max_length = max(max_length,len(v))
        # #不同类的动作组合，变量，取值，增减，倍率
        #取值： 0 不取值，做后续动作 1 随机取常量列表中的值 2从范围内随机取值
        #增减 0 减小 1 增加
        #倍率 0 1 2 3 4 5 6 7 8    10的次方
        self.actions = get_actions(torch.arange(0, 10))

        self.actions.to(device)
        # self.variables = {index: item for index, item in enumerate(self.variables)}
        # self.action_space = BoxActionSpace([torch.tensor(0), torch.tensor(-10000)],
        #                                    [torch.tensor(len(self.variables)), torch.tensor(10000)])
        self.action_space = DiscreteActionSpace(self.actions)
        print('action_space')
        print(self.action_space)
        print(self.actions.shape)
        del self.actions
        torch.cuda.empty_cache()
        return self.state, self.action_space

    def handle_satisfiable(self,solver, time_out, reward, performance):
        reward += int(1 / time_out * 500 * 1000)
        performance += 1
        stats = solver.statistics()
        print("求解时间:", stats.get_key_value('time'))
        return reward, performance, True

    def handle_unknown(self,solver, time_out, reward, performance, smtlib_str):
        reward += -int(time_out / 10000) / 2
        # 表现重新计算
        # performance = 0
        return reward, performance, False

    def handle_case(self, r, solver, time_out, reward, performance):
        if r == z3.sat:
            return self.handle_satisfiable(solver, time_out, reward, performance)
        elif r == z3.unknown:
            return self.handle_unknown(solver, time_out, reward, performance, self.smtlib_str)
        else:
            return self.handle_unsatisfiable(time_out, reward, performance)
    def handle_unsatisfiable(self, time_out, reward, performance):
        reward += -int(time_out / 10000)
        #表现重新计算
        # performance = 0
        return reward, performance, False

    def action_space(self):
        """Returns the action space of the environment."""
        pass

    def process_text(self, text,variable_pred):
        text_limit = 120000
        client = OpenAI(
            base_url='https://svip.xty.app/v1',
            api_key=''
        )
        # Split the text into chunks of 4096 characters
        responses = []

        # Add 'system' role message before the loop
        system_message = {
            "role": "system",
            "content": "You are an advanced SAT/SMT solver, focusing on the optimization and resolution of logical constraint problems. Your input consists of two parts: first, the counterexamples of failed solution assignments previously chosen, and second, the strings in SMT-LIB format that needs to be solved. You should analyze these inputs, using logical reasoning and heuristic methods to determine which variable assignments led to the failure of the solution, and identify the variable assignments that satisfy all constraint conditions. For counterexample assignment of a variable, you should no longer select that value as output. The output should be a set of specific variable assignments that can satisfy all the constraints defined in the `.smt2` file. Your task is to find the specific values that should be assigned to the variables provided in the prompt to ensure that the entire constraint system is satisfiable.You should output only the numeric value"
        }

        chunks = [text[i:i + text_limit] for i in range(0, len(text), text_limit)]

        # '以上是我通过分段的方式给你的smt文本，你需要对其进行分析，然后为了使其求解加速得到sat结果，给出一个或者多个具体的变量名(VAR1,VAR2...)和其应该赋值的具体值。/n',
        for chunk in chunks:
            chat_completion = client.chat.completions.create(
                messages=[
                    system_message,  # Including the system role message here
                    {
                        "role": "user",
                        "content": chunk + f'This is the The variable values from the previous failed SAT solving attempt and SMT text given to you in segments; analyze it. To speed up the solution and obtain a SAT result, provide a specific number that {variable_pred} should be assigned to. However, do not choose the values that have already failed to solve. Output only the numeric value. Do not output any other text, explanations, or symbols. The output must be a single number.'
                    },
                ],
                # model="gpt-3.5-turbo",
                model="gpt-4o-mini",
                max_tokens = 5,

            )
            # Correct way to get the assistant's message
            # print(chat_completion.choices[0].message.content)
            responses.append(chat_completion.choices[0].message.content)

        print(responses)
        return responses
    def step(self, action):
        # print(action)
        self.step_count += 1
        try:
            reward = 0
            # variable_pred = self.variables[action]
            # action = self.action_space.
            action = self.action_space.actions_batch[action]
            print('当前动作:******************')
            print(action)
            action_v = int(action[0].item())
            if self.last_action is None:
                action_v = action_v % len(self.variables)
                variable_pred = self.variables[action_v]
            else:
                action_v = (self.last_action + action_v) % len(self.variables)
                variable_pred = self.variables[int(action_v.item())]
            print(variable_pred)
            # 在一次执行过程中，action不能重复
            if self.concrete_count == 0:
                if len(self.counterexamples_list) > 0 and len(self.counterexamples_list[-1]) == 0:
                    pass
                else:
                    self.counterexamples_list.append([])
            #判断变量有没有被赋值
            var_flag = -1
            for index, value in enumerate(self.counterexamples_list[-1]):
                if value[0] == variable_pred:
                    var_flag = index
            # 如果选择了第一个数，随机选择一个值
            ce_json = json.dumps(self.counterexamples_list)

            responses = self.process_text(ce_json + '\n' + self.smtlib_str,variable_pred)
            index = len(responses) - 1
            while index > 0 and is_number(responses[index]) == False:
                index -= 1
            selected_int = responses[index]
            with open('example.txt', 'a', encoding='utf-8') as file:
                # 将字符串写入文件
                file.write(ce_json +'\n' + self.smtlib_str + '\n' + selected_int + '\n')

            print('llm挑选的具体值')
            print(selected_int)
            type_info = find_var_declaration_in_string(self.smtlib_str_original, variable_pred)
            print(type_info)
            print(type(type_info))
            if 'BitVec' in type_info:
                type_scale = type_info.split(' ')[-1]
                print(type_scale)
                new_constraint = "(assert (= {} (_ bv{} {})))\n".format(variable_pred, str(selected_int), type_scale)
            elif type_info in ['Int', 'Real']:
                new_constraint = "(assert (= {} {}))\n".format(variable_pred, str(selected_int))
                type_scale = 0
            related_assertions = self.v_related_assertions[variable_pred]
            count = 0
            if len(related_assertions) > 0:

                # related_assertions = find_assertions_related_to_var_name(assertions, variable_pred)

                for a in related_assertions:
                    solver_related = Solver()
                    solver_related.add(a)
                    smtlib_str_before, smtlib_str_after = split_at_check_sat(solver_related.to_smt2())

                    new_smtlib_str = smtlib_str_before + new_constraint + smtlib_str_after
                    solver_related = Solver()
                    assertions = parse_smt2_string(new_smtlib_str)
                    for a in assertions:
                        solver_related.add(a)
                    time_out = 10000
                    solver_related.set("timeout", time_out)
                    r = solver_related.check()
                    if z3.sat == r:
                        count += 1
                        reward += 5

                    # elif z3.unknown == r:
                    #     reward += int(1 / time_out * 1000)
                    #     self.used_variables.append(variable_pred)
                    #     self.concrete_count += 1
                    # else:
                    #     reward += -int(time_out / 10000)
                    #     self.used_variables.append(variable_pred)
                    #     self.concrete_count += 1
            print("约束个数和通过的个数")
            print(count, len(related_assertions))
            if count == len(related_assertions):
                if variable_pred not in self.used_variables:
                    self.used_variables.append(variable_pred)
                    self.concrete_count += 1
                    self.counterexamples_list[-1].append([variable_pred, selected_int])
                    smtlib_str_before, smtlib_str_after = split_at_check_sat(self.smtlib_str)
                    # new_constraint = "(assert (= {} (_ bv{} {})))\n".format(variable_pred, selected_int, type_scale)
                    self.smtlib_str = smtlib_str_before + new_constraint + smtlib_str_after
                else:
                    for index, value in enumerate(self.counterexamples_list[-1]):
                        if value[0] == variable_pred:  # 假设我们要根据某个条件来更新元素
                            last_ce = copy.deepcopy(self.counterexamples_list[-1])
                            last_ce[index] = [variable_pred, selected_int]
                            self.counterexamples_list.append(last_ce)
                    # 此次具体化不记入，而是更新
                    # self.concrete_count -= 1
                    # if type_info in ['BV']:
                    self.smtlib_str = repalce_veriable(self.smtlib_str, variable_pred, selected_int, type_scale, type_info)
                    # elif type_info in ['Int']:
                    #     new_constraint = "(assert (= {} {}))\n".format(variable_pred, str(selected_int))

                assertions = parse_smt2_string(self.smtlib_str)
                solver = Solver()
                for a in assertions:
                    solver.add(a)
                reward += self.calculate_reward(solver)
                self.z3ast = solver.assertions()
                var_list = normalize_smt_str_without_replace(solver.to_smt2())
                self.state = self.embedder.get_max_pooling_embedding(solver.to_smt2(), var_list)

                #考虑需要修改的逻辑
                # if self.concrete_count == len(self.variables):
                #     self.concrete_finish = True
                #     self.reset()
            else:
                return ActionResult(
                    observation=self.state,
                    reward=float(reward),
                    terminated=self.finish,
                    truncated=self.finish,
                    info={},
                    available_action_space=self.action_space, )

            # 清除内存
            torch.cuda.empty_cache()
        except Exception as e:
            print(e)
            print('some problems are triggered')
            traceback.print_exc()
            # print(self.smtlib_str)
            self.state = self.state_original.clone().detach()
            reward = 0
        if self.step_count > 10000:
            self.finish = True
        return ActionResult(
            observation=self.state,
            reward=float(reward),
            terminated=self.finish,
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
        # solver.set("timeout", 60000)
        # 判断新产生的序列和之前有没有重复
        # 判断是否存在反例
        if len(self.counterexamples_list) > 1:
            if self.counterexamples_list[-1] in self.counterexamples_list[:len(self.counterexamples_list) - 1]:
                reward += -10
                self.counterexamples_list.pop()
                #出现反例
                return reward
            else:
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
        var_list = normalize_smt_str_without_replace(solver_part.to_smt2())
        new_state = self.embedder.get_max_pooling_embedding(solver_part.to_smt2(), var_list)
        output = self.predictor(new_state)
        predicted_solvability__part = (output > 0.5).int().item()
        if predicted_solvability__part == 1:
            reward += 5
            performance += 1

            output_time = self.predictor_time(new_state)
            _, predicted_time = torch.max(output_time, 1)
            print(int(predicted_time.item()))
            time_out = int(self.time_dict[int(predicted_time.item())] * 1000 * 1.2)

            solver_part.set("timeout", time_out)
            r = solver_part.check()
            reward, performance, finish = self.handle_case(r, solver_part, time_out, reward, performance)
        #即使预测不可解，也要继续
        var_list = normalize_smt_str_without_replace(self.smtlib_str)
        new_state = self.embedder.get_max_pooling_embedding(self.smtlib_str, var_list)
        output = self.predictor(new_state)
        predicted_solvability = (output > 0.5).int().item()
        if predicted_solvability == 1:
            reward += 5
            performance += 1
        #即使预测不可解，也要继续
        output_time = self.predictor_time(new_state)
        _, predicted_time = torch.max(output_time, 1)
        print(int(predicted_time.item()))
        time_out = int(self.time_dict[int(predicted_time.item())] * 1000 * 1.2)

        solver.set("timeout", time_out)
        r = solver.check()
        reward, performance, finish = self.handle_case(r, solver, time_out, reward, performance)
        if finish:
            stats = solver.statistics()
            reward += int(1 / time_out * 500 * 1000)
            performance += 1
            self.finish = True
            self.solve_time = stats.get_key_value('time')
            print("求解时间:", stats.get_key_value('time'))
        else:
            reward += -int(time_out / 10000)
        if performance < self.last_performance:
            self.reset()
        self.last_performance = performance
        return reward

    def merge_intervals(self, intervals):
        # 首先根据区间的起始点对区间进行排序
        intervals.sort(key=lambda x: x[0])

        # 初始化合并后的区间列表
        merged = []

        # 遍历排序后的区间列表
        for interval in intervals:
            # 如果合并列表为空，或者当前区间的起始点大于合并列表中最后一个区间的结束点
            if not merged or merged[-1][1] < interval[0]:
                # 直接添加当前区间到合并列表
                merged.append(interval)
            else:
                # 否则，合并当前区间与合并列表中最后一个区间
                merged[-1][1] = max(merged[-1][1], interval[1])

        return merged

    def divide_intervals(self, intervals, m):
        # 计算总的整数个数
        total_count = sum(end - start + 1 for start, end in intervals)

        if m >= total_count:
            return [(start, end) for start, end in intervals]

            # 计算小区间的数量 n
        n = total_count // m
        # 如果 m 不能整除总的整数数量，增加一个小区间
        if total_count % m != 0:
            n += 1
        # 计算每个小区间应该包含的整数个数
        if n == 0 or total_count == 0:
            return []

        # m = total_count // n
        # if m == 0:
        #     return [intervals]

        result = []
        current_interval = []
        current_count = 0

        for interval in intervals:
            start, end = interval
            length = end - start + 1

            while length > 0:
                if current_count + length <= m:
                    current_interval.append([start, end])
                    current_count += length
                    break
                else:
                    part_length = m - current_count
                    current_interval.append([start, start + part_length - 1])
                    result.append(current_interval)
                    start += part_length
                    length -= part_length
                    current_interval = []
                    current_count = 0

            if current_count == m:
                result.append(current_interval)
                current_interval = []
                current_count = 0

        if current_interval:
            result.append(current_interval)

        # 平均分配剩余的整数到各小区间
        remaining = total_count % n
        if remaining:
            index = 0
            for i in range(remaining):
                result[index].append([start + i, start + i])
                index = (index + 1) % n

        return result

    def divide_intervals_by_n(self, intervals, n):
        total_count = sum(end - start + 1 for start, end in intervals)

        # 计算每个小区间应该包含的整数个数
        if n == 0 or total_count == 0:
            return []

        m = total_count // n
        if m == 0:
            # 如果每个区间至少能分到一个整数，则返回原区间g
            return [intervals]

        result = []
        current_interval = []
        current_count = 0
        start_index = 0  # 用于记录当前处理到的起始点

        for interval in intervals:
            start, end = interval
            length = end - start + 1

            while length > 0:
                if current_count + length <= m:
                    current_interval.append([start, start + current_count + length - 1])
                    current_count += length
                    length = 0  # 已经完全放入当前区间，无需再减
                    start_index = start + current_count  # 更新起始点
                else:
                    part_length = m - current_count
                    current_interval.append([start, start + part_length - 1])
                    result.append(current_interval)
                    start += part_length
                    length -= part_length
                    current_count = 0
                    current_interval = []

            if current_count == m:
                result.append(current_interval)
                current_interval = []
                current_count = 0

        if current_interval:
            result.append(current_interval)

        # 计算剩余的整数数量
        remaining = total_count % n

        # 如果有剩余的整数，将它们通过区间的形式添加到一个新的列表中
        if remaining:
            # 找到最后一个区间的起始点
            last_start = intervals[-1][0] + start_index
            # 创建包含剩余整数的新区间列表
            remaining_intervals = [(last_start, last_start + remaining - 1)]
            result.append(remaining_intervals)

        return result

    def random_from_subinterval(self, subintervals, n):
        # 随机选择第n个小区间
        chosen_subinterval = subintervals[n - 1]

        # 计算这个小区间中包含的整数个数
        total_numbers = 0
        for start, end in chosen_subinterval:
            total_numbers += (end - start + 1)

        # 随机选择一个整数索引
        random_index = random.randint(1, total_numbers)

        # 初始化计数器
        count = 0
        for start, end in chosen_subinterval:
            # 如果随机索引在当前区间内
            if count + (end - start + 1) >= random_index:
                # 返回对应的整数
                return start + (random_index - count) - 1
            count += (end - start + 1)  # 更新计数器

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

    def observation_space(self) -> Space:
        """Returns the observation space of the environment."""
        pass


def visit(expr, variables):
    if is_const(expr) and expr.decl().kind() == Z3_OP_UNINTERPRETED:
        # Add only uninterpreted functions (which represent variables)
        # print(type(self.variables))
        variables.add(str(expr))
    else:
        # Recursively visit children for composite expressions
        for child in expr.children():
            visit(child, variables)


#不同类的动作组合，变量，取值，增减，倍率
def get_actions(tensor_1d_1):

    result_tensor = tensor_1d_1
    # 确保结果张量在正确的设备上
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    result_tensor = result_tensor.to(device)

    # 清理缓存
    torch.cuda.empty_cache()

    # 打印结果以供调试
    print('*********************')
    print(result_tensor, type(result_tensor))

    return result_tensor


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
