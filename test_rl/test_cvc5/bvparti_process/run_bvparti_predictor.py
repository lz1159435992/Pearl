#!/usr/bin/env python3
"""
BVParti Predictor Runner
基于STP-Parti-Bitwuzla求解器的强化学习预测器
仿照cvc5_process/run_predictor.py的架构，但使用BVParti作为底层求解器
"""

import math
import os
import datetime
from multiprocessing import Process, Manager, Queue, set_start_method

# 设置多进程启动方法为'spawn'，解决CUDA在子进程中的初始化问题
try:
    set_start_method('spawn')
except RuntimeError:
    # 方法已被设置，忽略错误
    pass

# 设置环境变量
os.environ['ALL_PROXY'] = ''
os.environ['all_proxy'] = ''

import signal
import sys
import argparse
import ast
import json
import random
import re
import time
import copy
import traceback
import tempfile
import subprocess
import string
import shutil
from decimal import Decimal, getcontext
from loguru import logger

# 核心依赖
import torch
import numpy as np
from z3 import *
from z3.z3 import parse_smt2_string, Solver as Z3_Solver, sat, unknown, unsat
from openai import OpenAI
from ollama import Client

# Pearl框架组件
from pearl.policy_learners.sequential_decision_making.soft_actor_critic import SoftActorCritic
from pearl.replay_buffers.sequential_decision_making.bootstrap_replay_buffer import FIFOOffPolicyReplayBuffer
from pearl.utils.functional_utils.experimentation.set_seed import set_seed
from pearl.action_representation_modules.identity_action_representation_module import IdentityActionRepresentationModule
from pearl.history_summarization_modules.lstm_history_summarization_module import LSTMHistorySummarizationModule
from pearl.api import Space
from pearl.api.action_result import ActionResult
from pearl.api.environment import Environment
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace
from pearl.pearl_agent import PearlAgent

# 导入兼容的convert_timeout_to_unknown函数
from test_rl.test_cvc5.bvparti_process.test_group_get_dis_smt_comp_bert_embeding_single import (
    convert_timeout_to_unknown, get_solve_result_and_time
)

# 导入工具函数
from test_rl.test_script.utils import (
    parse_smt2_in_parts, process_smt_lib_string, fetch_data_as_dict,
    solve_and_measure_time, model_to_dict, load_dictionary, extract_variables_from_smt2_content,
    normalize_variables, normalize_smt_str, MyException, timeout_handler, setup_logger,
    find_var_declaration_in_string, split_at_check_sat, find_assertions_related_to_var_name,
    find_assertions_related_to_var_names_optimized, repalce_veriable, normalize_smt_str_without_replace,
    solve_assertion_get_range
)

# 导入模型和嵌入器
from test_rl.bert_embedder_test import CodeEmbedder_normalize
from train_predictor import EnhancedEightClassModel, SimpleClassifier
from test_rl.test_script.online_learning_break import online_learning

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def is_number(s):
    pattern = r'^(\d+|\d+\.\d+|\d+\/\d+)$'
    return re.match(pattern, s) is not None

def visit(expr, variables):
    """访问Z3表达式并收集变量"""
    if is_const(expr) and expr.decl().kind() == Z3_OP_UNINTERPRETED:
        variables.add(str(expr))
    else:
        for child in expr.children():
            visit(child, variables)

def get_actions(tensor_1d_1):
    """获取动作空间"""
    result_tensor = tensor_1d_1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    result_tensor = result_tensor.to(device)
    torch.cuda.empty_cache()
    return result_tensor

# BVParti求解器配置
class BVPartiConfig:
    def __init__(self):
        # 获取AriParti_sync目录下的STP-Parti-Bitwuzla求解器路径
        self.ariparti_base_path = "/home/lz/PycharmProjects/Pearl/test_rl/AriParti_sync"
        self.bvparti_solver_path = os.path.join(
            self.ariparti_base_path,
            "STP-Parti-Bitwuzla-at-SMT-COMP-2025-build/solver"
        )

        # 各组件路径
        self.bvparti_bin = os.path.join(self.bvparti_solver_path, "BVPartition-bin")
        self.partitioner_bin = os.path.join(self.bvparti_solver_path, "partitioner-bin")
        self.bitwuzla_bin = os.path.join(self.bvparti_solver_path, "bitwuzla-0.8.0-bin")
        self.run_bvparti_script = os.path.join(self.bvparti_solver_path, "run_BVParti.py")

        # 默认参数
        self.max_running_tasks = 8
        self.time_limit = 1200
        self.temp_dir_base = "/tmp/bvp-files"

        # 验证路径存在性
        self._validate_paths()

    def _validate_paths(self):
        """验证所有必要的路径是否存在"""
        paths_to_check = [
            self.bvparti_bin,
            self.partitioner_bin,
            self.bitwuzla_bin,
            self.run_bvparti_script
        ]

        for path in paths_to_check:
            if not os.path.exists(path):
                logger.warning(f"Path does not exist: {path}")

# 求解器结果类
class SolverResult:
    def __init__(self, solve_time, result, model):
        self.solve_time = solve_time
        self.result = result
        self.model = model

# 求解器基类
class Solver:
    def solve(self, smtlib_str, timeout=5):
        raise NotImplementedError

# BVParti求解器包装类
class BVPartiSolver(Solver):
    def __init__(self, config=None):
        self.config = config or BVPartiConfig()
        self.temp_folders = []

    def generate_random_string(self, length=16):
        """生成随机字符串用于临时文件夹名"""
        characters = string.ascii_letters + string.digits
        return ''.join(random.choice(characters) for _ in range(length))

    def prepare_temp_folder(self):
        """准备临时文件夹"""
        temp_folder_name = self.generate_random_string(16)
        temp_folder_path = os.path.join(self.config.temp_dir_base, temp_folder_name)
        os.makedirs(temp_folder_path, exist_ok=True)
        self.temp_folders.append(temp_folder_path)
        return temp_folder_path

    def solve(self, smtlib_str, timeout=5):
        """
        使用BVParti求解SMT字符串

        Args:
            smtlib_str: SMT-LIB格式的字符串
            timeout: 超时时间（秒）

        Returns:
            SolverResult: 求解结果对象
        """
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.smt2', delete=False) as f:
            f.write(smtlib_str)
            f.flush()
            temp_file_path = f.name

        try:
            result, execution_time, stdout, stderr = self.solve_smt2_file(temp_file_path, timeout)
            return SolverResult(execution_time, result, stdout)
        finally:
            # 清理临时文件
            try:
                os.unlink(temp_file_path)
            except:
                pass

    def solve_smt2_file(self, smt2_file_path, timeout=None):
        """
        使用BVParti求解SMT2文件

        Args:
            smt2_file_path: SMT2文件路径
            timeout: 超时时间（秒），默认使用配置中的time_limit

        Returns:
            tuple: (result, execution_time, stdout, stderr)
        """
        if timeout is None:
            timeout = self.config.time_limit

        temp_folder_path = self.prepare_temp_folder()

        try:
            start_time = time.time()

            # 构建命令
            cmd = [
                self.config.bvparti_bin,
                '--file', smt2_file_path,
                '--output-dir', temp_folder_path,
                '--partitioner', self.config.partitioner_bin,
                '--solver', self.config.bitwuzla_bin,
                '--max-running-tasks', str(self.config.max_running_tasks),
                '--time-limit', str(timeout)
            ]

            logger.debug(f"Running BVParti command: {' '.join(cmd)}")

            # 执行命令
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout + 10,  # 给额外的10秒缓冲时间
                text=True
            )

            end_time = time.time()
            execution_time = end_time - start_time

            # 解析结果
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            # 从stdout中提取求解结果
            if 'sat' in stdout.lower():
                if 'unsat' in stdout.lower():
                    solve_result = 'unsat'
                else:
                    solve_result = 'sat'
            elif 'unsat' in stdout.lower():
                solve_result = 'unsat'
            elif 'timeout' in stdout.lower() or 'unknown' in stdout.lower():
                solve_result = 'timeout'
            else:
                solve_result = 'unknown'

            return solve_result, execution_time, stdout, stderr

        except subprocess.TimeoutExpired:
            return 'timeout', timeout, '', 'Process timed out'
        except Exception as e:
            logger.error(f"Error running BVParti: {e}")
            return 'error', 0, '', str(e)
        finally:
            # 清理临时文件夹
            self.cleanup_temp_folder(temp_folder_path)

    def cleanup_temp_folder(self, temp_folder_path):
        """清理临时文件夹"""
        try:
            if os.path.exists(temp_folder_path):
                shutil.rmtree(temp_folder_path, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp folder {temp_folder_path}: {e}")

    def cleanup_all_temp_folders(self):
        """清理所有临时文件夹"""
        for temp_folder in self.temp_folders:
            self.cleanup_temp_folder(temp_folder)
        self.temp_folders.clear()

# 其他求解器类（保持兼容性）
class Z3Solver(Solver):
    def solve(self, smtlib_str, timeout=5):
        start = time.time()
        try:
            s = Z3_Solver()
            s.set("timeout", int(timeout * 1000))  # timeout in milliseconds
            try:
                z3_exprs = parse_smt2_string(smtlib_str)
                if isinstance(z3_exprs, list):
                    s.add(*z3_exprs)
                else:
                    s.add(z3_exprs)
            except Exception as e:
                elapsed = time.time() - start
                return SolverResult(elapsed, 'parse_error', str(e))
            result = None
            model = None
            check_result = s.check()
            elapsed = time.time() - start
            if check_result == sat:
                result = 'sat'
                model = s.model().sexpr()
            elif check_result == unsat:
                result = 'unsat'
                model = None
            elif check_result == unknown:
                result = 'unknown'
                model = s.reason_unknown()
            else:
                result = str(check_result)
                model = None
        except Exception as e:
            elapsed = time.time() - start
            result = 'error'
            model = str(e)
        return SolverResult(elapsed, result, model)

class CVC5Solver(Solver):
    def solve(self, smtlib_str, timeout=5):
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.smt2', delete=False) as f:
            f.write(smtlib_str)
            f.flush()
            # CVC5使用毫秒为单位
            cmd = ['cvc5', '--lang', 'smt2', '--produce-models', f.name, f'--tlimit={int(timeout*1000)}']
            logger.info(f"Running command: {' '.join(cmd)}")
            start = time.time()
            try:
                # 进程超时设置比求解器超时稍长一点，确保求解器有足够时间响应
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout+1)
                elapsed = time.time() - start
                output = proc.stdout
                if 'unsat' in output:
                    result = 'unsat'
                    model = None
                elif 'sat' in output:
                    result = 'sat'
                    model = output
                else:
                    result = 'unknown'
                    model = output
            except subprocess.TimeoutExpired:
                elapsed = time.time() - start
                result = 'timeout'
                model = None
            finally:
                try:
                    os.unlink(f.name)  # 清理临时文件
                except:
                    pass
        return SolverResult(elapsed, result, model)

def get_solver(solver_name):
    """获取指定的求解器实例"""
    solver_map = {
        'z3': Z3Solver(),
        'cvc5': CVC5Solver(),
        'bvparti': BVPartiSolver(),
    }
    return solver_map.get(solver_name.lower(), BVPartiSolver())  # 默认使用BVParti求解器

class ConstraintSimplificationEnv_test(Environment):
    def __init__(self, embedder, z3ast, model, model_time, smtlib_str, file_path, var_dict, constant_list, solver_name='bvparti', llm_host='http://172.29.7.221:32903', llm_model='llama3.1:70b'):
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
        # 确保反例列表正确初始化为包含一个空列表
        self.counterexamples_list = [[]]
        self.finish = False
        self.used_variables = []
        self.state_count = 0
        self.predictor = model
        self.predictor_time = model_time
        self.last_performance = 0
        self.solve_time = 0  # 最终成功求解时间
        self.total_solve_time = 0  # 累计所有求解操作的时间
        self.llm_time = 0  # 累计所有LLM调用的时间
        self.solver_name = solver_name
        self.solver = get_solver(solver_name)  # 获取指定的求解器实例
        self.v_related_assertions, self.var_range_dict = solve_assertion_get_range(self.z3ast, self.variables)
        self.var_range_dict_n = {}
        self.range_init()
        self.time_dict = {
            0: 20,
            1: 1,
            2: 20,
            3: 50,
            4: 100,
            5: 200,
            6: 500,
            7: 1000,
        }
        # 保存LLM配置
        self.llm_host = llm_host
        self.llm_model = llm_model
        # 记录初始化完成
        logger.info(f"环境实例初始化完成: 文件={file_path}, 变量数量={len(self.variables)}, LLM时间={self.llm_time}秒, 反例列表={self.counterexamples_list}")

    def range_init(self):
        for variable in self.variables:
            type_info = find_var_declaration_in_string(self.smtlib_str_original, variable)
            if 'BitVec' in type_info:
                type_scale = type_info.split(' ')[-1]
                max_value = 2 ** int(type_scale) - 1
                min_value = 0
            elif type_info in ['Int', 'Real']:
                max_value = 2147483650
                min_value = -2147483650
                type_scale = 0
            self.var_range_dict[variable].append([min_value, max_value])

    def reset(self, seed=None):
        self.concrete_finish = False
        self.concrete_count = 0
        self.used_variables = []
        self.state = self.state_original.clone().detach()
        self.z3ast = copy.deepcopy(self.z3ast_original)
        self.smtlib_str = copy.deepcopy(self.smtlib_str_original)
        self.last_performance = 0

        # 注意：不重置llm_time和counterexamples_list等需要持续存储的数据
        # 这些数据应该只在__init__中初始化，在整个处理过程中保持累积

        self.actions = get_actions(torch.arange(0, len(self.variables)))
        self.actions.to(device)
        self.action_space = DiscreteActionSpace(self.actions)
        del self.actions
        torch.cuda.empty_cache()
        logger.info(f"环境已重置：状态和动作空间已更新，LLM时间保持为{self.llm_time:.3f}秒, 反例列表长度={len(self.counterexamples_list)}")
        return self.state, self.action_space

    def handle_case(self, r, solver_result, time_out, reward, performance):
        if r == 'sat':
            return self.handle_satisfiable(solver_result, time_out, reward, performance)
        elif r == 'unknown' or r == 'timeout':
            return self.handle_unknown(solver_result, time_out, reward, performance)
        else:
            return self.handle_unsatisfiable(time_out, reward, performance, solver_result)

    def handle_satisfiable(self, solver_result, time_out, reward, performance):
        reward += int(1 / time_out * 500 * 1000)
        performance += 1
        # 记录日志
        logger.info(f"求解结果: sat, 本次求解耗时: {solver_result.solve_time:.3f}秒")
        return reward, performance, True

    def handle_unknown(self, solver_result, time_out, reward, performance):
        reward += -int(time_out / 10000) / 2
        # 记录日志
        logger.info(f"求解结果: unknown, 本次求解耗时: {solver_result.solve_time:.3f}秒")
        return reward, performance, False

    def handle_unsatisfiable(self, time_out, reward, performance, solver_result=None):
        reward += -int(time_out / 10000)
        # 记录日志
        if solver_result is not None:
            logger.info(f"求解结果: unsat, 本次求解耗时: {solver_result.solve_time:.3f}秒")
        return reward, performance, False

    def process_text_python(self, text, variable_pred):
        variables = self.variables
        var_str = ','.join(variables)

        system_message = {
            "role": "system",
            "content": """You are an advanced SAT/SMT solver, focusing on the optimization and resolution of logical constraint problems.
            Your input consists of two parts: first, the counterexamples of failed solution assignments previously chosen, and second, the strings in SMT-LIB format that needs to be solved.
            You should analyze these inputs, using logical reasoning and heuristic methods to determine which variable assignments led to the failure of the solution,
            and identify the variable assignments that satisfy all constraint conditions. The output should be a specific value assignment for the target variable that can satisfy all the constraints defined in the strings.
            Your task is to find the specific values that should be assigned to the variables provided in the prompt to ensure that the entire constraint system is satisfiable.You should output only the numeric value,
            with an example as follows: <value> . Do not output any other text, explanations, or symbols."""}
        user_message = {
            "role": "user",
            "content": text + f'This is the variable values from the previous failed SAT solving attempt and SMT text given to you in segments; analyze it. To speed up the solution and obtain a SAT result, provide a specific number that {variable_pred} should be assigned to. However, do not choose the values that have already failed to solve. Output only the numeric value. Do not output any other text, explanations, or symbols. The output must be a single number.'
        }

        # 记录LLM调用开始时间
        llm_start_time = time.time()

        # 使用配置的LLM主机和模型
        client = Client(host=self.llm_host)
        response = client.chat(
            model=self.llm_model,
            messages=[system_message, user_message],
            options={"temperature": 1},
            stream=True,
        )
        responses = []
        for chunk in response:
            responses.append(chunk['message']['content'])

        # 计算并累加LLM调用时间
        llm_elapsed_time = time.time() - llm_start_time
        self.llm_time += llm_elapsed_time

        # 记录日志
        logger.info(f"LLM调用耗时: {llm_elapsed_time:.3f}秒，累计LLM时间: {self.llm_time:.3f}秒")

        return responses

    def step(self, action):
        self.step_count += 1
        try:
            reward = 0
            action = self.action_space.actions_batch[action]
            action_v = action[0]
            variable_pred = self.variables[int(action_v.item())]

            if self.concrete_count == 0:
                if len(self.counterexamples_list) > 0 and len(self.counterexamples_list[-1]) == 0:
                    pass
                else:
                    self.counterexamples_list.append([])

            ce_json = json.dumps(self.counterexamples_list)
            text = "Here is the  counterexamples of failed solution assignments previously chosen in json formats:\n" + ce_json + "\n" \
                   + 'Here is the SMT file content:\n' + self.smtlib_str

            responses = self.process_text_python(text, variable_pred)
            index = len(responses) - 1
            while index > 0 and is_number(responses[index]) == False:
                index -= 1
            selected_int = responses[index]

            if int(selected_int) < self.var_range_dict[variable_pred][0][0] or int(selected_int) > \
                    self.var_range_dict[variable_pred][0][1]:
                self.counterexamples_list[-1].append([variable_pred, selected_int])
                self.reset()

            type_info = find_var_declaration_in_string(self.smtlib_str_original, variable_pred)
            if 'BitVec' in type_info:
                type_scale = type_info.split(' ')[-1]
                new_constraint = "(assert (= {} (_ bv{} {})))\n".format(variable_pred, str(selected_int), type_scale)
            elif type_info in ['Int', 'Real']:
                new_constraint = "(assert (= {} {}))\n".format(variable_pred, str(selected_int))
                type_scale = 0

            related_assertions = self.v_related_assertions[variable_pred]
            count = 0
            if len(related_assertions) > 0:
                for a in related_assertions:
                    # 创建z3 Solver对象并添加断言
                    z3_solver = Z3_Solver()
                    z3_solver.add(a)

                    # 将z3 assertion转换为标准SMT-LIB格式
                    smtlib_str_before, smtlib_str_after = split_at_check_sat(z3_solver.to_smt2())
                    new_smtlib_str = smtlib_str_before + new_constraint + smtlib_str_after

                    # 使用配置的求解器进行求解
                    solver_result = self.solver.solve(new_smtlib_str, timeout=10)
                    self.total_solve_time += solver_result.solve_time
                    if solver_result.result == 'sat':
                        count += 1
                        reward += 5
                        logger.info(f"变量 {variable_pred} 的相关断言求解成功：sat，耗时: {solver_result.solve_time:.3f}秒，当前累计求解时间: {self.total_solve_time:.3f}秒")
                    else:
                        logger.info(f"变量 {variable_pred} 的相关断言求解失败: {solver_result.result}，耗时: {solver_result.solve_time:.3f}秒，当前累计求解时间: {self.total_solve_time:.3f}秒")

            if count == len(related_assertions):
                if variable_pred not in self.used_variables:
                    self.used_variables.append(variable_pred)
                    self.concrete_count += 1
                    self.counterexamples_list[-1].append([variable_pred, selected_int])
                    smtlib_str_before, smtlib_str_after = split_at_check_sat(self.smtlib_str)
                    self.smtlib_str = smtlib_str_before + new_constraint + smtlib_str_after
                else:
                    for index, value in enumerate(self.counterexamples_list[-1]):
                        if value[0] == variable_pred:
                            last_ce = copy.deepcopy(self.counterexamples_list[-1])
                            last_ce[index] = [variable_pred, selected_int]
                            self.counterexamples_list.append(last_ce)
                    self.smtlib_str = repalce_veriable(self.smtlib_str, variable_pred, selected_int, type_scale, type_info)

                # 不提前执行求解过程，仅解析约束，类似原始代码的处理方式
                self.z3ast = parse_smt2_string(self.smtlib_str)
                reward += self.calculate_reward(self.solver)

                # 更新状态
                var_list = normalize_smt_str_without_replace(self.smtlib_str)
                self.state = self.embedder.get_max_pooling_embedding(self.smtlib_str, var_list)

            else:
                return ActionResult(
                    observation=self.state,
                    reward=float(reward),
                    terminated=self.finish,
                    truncated=self.finish,
                    info={},
                    available_action_space=self.action_space,)

            del action
            del action_v
            torch.cuda.empty_cache()
        except MyException as e:
            raise MyException("Timeout!")
        except Exception as e:
            print(e)
            traceback.print_exc()
            self.state = self.state_original.clone().detach()
            reward = 0

        if self.step_count > 50000000000:
            self.finish = True

        return ActionResult(
            observation=self.state,
            reward=float(reward),
            terminated=self.finish,
            truncated=self.finish,
            info={},
            available_action_space=self.action_space,)

    def calculate_reward(self, solver):
        performance = 0
        reward = 0
        count = 0

        if len(self.counterexamples_list) > 1:
            if self.counterexamples_list[-1] in self.counterexamples_list[:len(self.counterexamples_list) - 1]:
                reward += -10
                self.counterexamples_list.pop()
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

        # 创建部分solver进行初步检查
        solver_part = Solver()
        assertions = parse_smt2_string(self.smtlib_str)

        assertions_list = []
        for a in assertions:
            assertions_list.append(a)

        # 随机采样一半的断言进行验证
        indexes = random.sample(range(len(assertions_list)), int(len(assertions_list) * 0.5))
        res = [assertions_list[i] for i in sorted(indexes)]

        # 转换为SMTLIB格式并交由自定义求解器处理
        z3_solver_part = Z3_Solver()
        for r in res:
            z3_solver_part.add(r)

        var_list = normalize_smt_str_without_replace(z3_solver_part.to_smt2())
        new_state = self.embedder.get_max_pooling_embedding(z3_solver_part.to_smt2(), var_list)
        output = self.predictor(new_state)
        predicted_solvability__part = (output > 0.5).int().item()

        if predicted_solvability__part == 1:
            reward += 5
            performance += 1

            output_time = self.predictor_time(new_state)
            _, predicted_time = torch.max(output_time, 1)
            print(int(predicted_time.item()))
            # 与原始代码保持一致，使用毫秒单位
            time_out = int(self.time_dict[int(predicted_time.item())] * 1000 * 1.2)

            # 使用自定义求解器，处理毫秒为单位的timeout
            solver_result_part = solver.solve(z3_solver_part.to_smt2(), timeout=time_out/1000)
            # 记录求解信息
            self.total_solve_time += solver_result_part.solve_time
            logger.info(
                f"part约束求解结果: {solver_result_part.result}，耗时: {solver_result_part.solve_time:.3f}秒，当前累计求解时间: {self.total_solve_time:.3f}秒")
            reward, performance, finish = self.handle_case(solver_result_part.result, solver_result_part, time_out, reward, performance)

        # 即使预测不可解，也要继续 - 处理完整约束
        var_list = normalize_smt_str_without_replace(self.smtlib_str)
        new_state = self.embedder.get_max_pooling_embedding(self.smtlib_str, var_list)
        output = self.predictor(new_state)
        predicted_solvability = (output > 0.5).int().item()
        if predicted_solvability == 1:
            reward += 5
            performance += 1

        # 即使预测不可解，也要继续
        output_time = self.predictor_time(new_state)
        _, predicted_time = torch.max(output_time, 1)
        print(int(predicted_time.item()))
        # 与原始代码保持一致，使用毫秒单位
        time_out = int(self.time_dict[int(predicted_time.item())] * 1000 * 1.2)

        # 使用自定义求解器，处理毫秒为单位的timeout - 这里才是真正的求解过程
        solver_result = solver.solve(self.smtlib_str, timeout=time_out/1000)
        # 记录求解信息
        self.total_solve_time += solver_result.solve_time
        logger.info(f"完整约束求解结果: {solver_result.result}，耗时: {solver_result.solve_time:.3f}秒，当前累计求解时间: {self.total_solve_time:.3f}秒")

        reward, performance, finish = self.handle_case(solver_result.result, solver_result, time_out, reward, performance)

        if finish:
            self.finish = True
            # 记录最终成功求解的时间，而不是总求解时间
            self.solve_time = solver_result.solve_time
            logger.info(f"约束已成功求解! 最终求解耗时: {self.solve_time:.3f}秒, 当前累计求解时间: {self.total_solve_time:.3f}秒")
        else:
            reward += -int(time_out*1000 / 10000)

        if performance < self.last_performance:
            self.reset()
        self.last_performance = performance
        return reward

    @staticmethod
    def counter_reward_function(total_length, unique_count):
        R_positive = 1
        R_negative = -1
        alpha = 1 / math.sqrt(total_length) if total_length > 0 else 1

        if unique_count > 0:
            reward = R_positive / math.log(1 + total_length) * 10
        else:
            reward = R_negative * alpha * 10

        return reward

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None

    def action_space(self):
        """Returns the action space of the environment."""
        pass

    def observation_space(self) -> Space:
        pass

    def set_model(self, model_name):
        """
        更改当前使用的LLM模型

        Args:
            model_name: 新的模型名称
        """
        self.llm_model = model_name
        logger.info(f"LLM模型已更改为: {model_name}")

def run_predictor(args):
    """
    运行预测器的主函数
    """
    # 初始化主进程的日志记录器，获取batch_id
    logger, batch_id = setup_logger()
    logger.info("=" * 80)
    logger.info(f"BVParti SMT约束求解预测器启动 - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"使用求解器: {args.solver}, LLM模型: {args.llm_model}, LLM服务器: {args.llm_host}")
    logger.info("=" * 80)

    # 将batch_id添加到args中，以便传递给子进程
    args.batch_id = batch_id

    # 加载RL字典
    with open(args.rl_dict_path, 'r') as file:
        rl_dict = json.load(file)
    logger.info(f"已加载RL字典，包含 {len(rl_dict)} 个条目")

    # 创建或加载信息字典
    if not os.path.exists(args.info_dict_path):
        info_dict = {}
        with open(args.info_dict_path, 'w') as file:
            json.dump(info_dict, file, indent=4)
        logger.info(f'文件{args.info_dict_path} 已创建。')
    else:
        info_dict = load_dictionary(args.info_dict_path)
        logger.info(f'已加载现有信息字典，包含 {len(info_dict)} 个条目')

    # 加载结果字典
    with open(args.result_dict_path, 'r') as file:
        result_dict = json.load(file)
    result_dict = convert_timeout_to_unknown(result_dict)

    # 获取实际的结果条目数量，兼容新旧格式
    results_count = len(result_dict.get("results", result_dict)) if "results" in result_dict else len(result_dict)
    logger.info(f"已加载结果字典，包含 {results_count} 个条目")

    # 处理每个结果
    file_count = 0
    success_count = 0
    failed_count = 0

    # 获取实际的结果数据，兼容新旧格式
    results_data = result_dict.get("results", result_dict) if "results" in result_dict else result_dict

    # 计算需要处理的文件数量，使用兼容的函数
    total_files = 0
    for key in results_data.keys():
        category, time_value = get_solve_result_and_time(result_dict, key)
        if category in ["sat", "unknown"] and time_value > args.time_threshold and key in rl_dict.keys() and key not in info_dict.keys():
            total_files += 1

    logger.info(f"总共需要处理 {total_files} 个文件")
    logger.info("=" * 80)

    # 创建键的列表并随机打乱顺序
    result_dict_keys = list(results_data.keys())
    random.shuffle(result_dict_keys)

    # 使用打乱后的键列表来遍历字典
    for key in result_dict_keys:
        category, time_value = get_solve_result_and_time(result_dict, key)
        if category in ["sat", "unknown"] and time_value > args.time_threshold and key in rl_dict.keys() and key not in info_dict.keys():
            file_count += 1
            logger.info("=" * 80)
            logger.info(f'开始处理文件 {file_count}/{total_files}: {key}')
            logger.info(f'原始求解结果: {category}, 原始求解时间: {time_value}秒')
            logger.info("-" * 50)

            # 清理GPU内存，确保环境隔离
            torch.cuda.empty_cache()

            # 处理文件
            try:
                old_info = info_dict.get(key, None)
                # 构造兼容的list1格式
                list1 = [category, time_value, 0]  # 第三个元素是内存使用，设为0
                process_single_file_with_timeout(key, list1, info_dict, args)
                # 检查处理结果
                new_info = info_dict.get(key, None)
                if new_info:
                    if new_info[7] == 'succeed':
                        success_count += 1
                        logger.info(f"文件处理成功: {key}")
                    else:
                        failed_count += 1
                        logger.info(f"文件处理失败: {key}")

                    # 检查LLM时间和反例
                    if new_info[6] == 0:
                        logger.warning(f"警告: {key} 的LLM时间为0")
                    if not new_info[-1] or len(new_info[-1]) == 0:
                        logger.warning(f"警告: {key} 的反例列表为空")
            except Exception as e:
                logger.error(f"处理文件 {key} 时出现异常: {str(e)}")
                failed_count += 1
                traceback.print_exc()

            logger.info(f'完成文件 {file_count}/{total_files}: {key}')
            logger.info("=" * 80)

    # 打印总结统计
    logger.info("\n" + "=" * 80)
    logger.info(f"处理完成 - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"总共处理: {file_count} 个文件")
    logger.info(f"成功: {success_count} 个")
    logger.info(f"失败: {failed_count} 个")
    logger.info("=" * 80)

def process_single_file_with_timeout(file_path, list1, info_dict, args):
    """
    使用多进程方式处理单个文件，带有超时控制

    Args:
        file_path: SMT文件路径
        list1: 结果列表
        info_dict: 信息字典
        args: 命令行参数
    """
    # 使用Manager共享数据
    manager = Manager()
    result_dict = manager.dict()

    # 创建共享队列，用于传递环境数据
    env_data_queue = Queue()

    # 创建子进程执行求解
    p = Process(target=_process_worker, args=(file_path, list1, result_dict, env_data_queue, args))
    p.start()

    # 设置默认值，以防获取不到环境数据
    env_data = {
        'total_solve_time': 0,
        'final_solve_time': 0,
        'llm_time': 0,
        'counterexamples_list': [[]],
        'start_time': time.time()  # 添加开始时间
    }

    # 每1秒检查一次环境数据队列，获取最新数据
    elapsed = 0
    check_interval = 1  # 1秒检查一次
    while elapsed < args.timeout:
        if not p.is_alive():
            break

        current_time = time.time()
        elapsed = current_time - env_data['start_time']

        # 不阻塞地检查队列
        try:
            while not env_data_queue.empty():
                new_env_data = env_data_queue.get_nowait()
                # 保持开始时间不变
                start_time = env_data['start_time']
                env_data.update(new_env_data)
                env_data['start_time'] = start_time
                logger.info(f"收到环境数据更新: 总求解时间={env_data['total_solve_time']:.3f}秒, "
                          f"最终求解时间={env_data['final_solve_time']:.3f}秒, "
                          f"LLM时间={env_data['llm_time']:.3f}秒, "
                          f"反例数量={len(env_data['counterexamples_list'])}, "
                          f"已执行时间={elapsed:.3f}秒")
        except Exception as e:
            logger.warning(f"获取环境数据时出错: {str(e)}")

        time.sleep(check_interval)

    # 检查进程是否仍在运行
    if p.is_alive():
        logger.warning(f"处理文件 {file_path} 超时，已强制终止")

        # 在终止前再次尝试获取最新环境数据
        try:
            while not env_data_queue.empty():
                new_env_data = env_data_queue.get_nowait()
                env_data.update(new_env_data)
        except:
            pass

        # 终止进程
        p.terminate()
        p.join()

        # 设置超时结果，但使用已收集的环境数据
        result_list = [list1[0], list1[1], list1[2]]# 使用原始结果作为基础
        result_list.append(args.timeout)  # 总执行时间为超时时间
        result_list.append(env_data['total_solve_time'])  # 使用收集到的总求解时间
        result_list.append(env_data['final_solve_time'])  # 使用收集到的最终求解时间
        result_list.append(env_data['llm_time'])  # 使用收集到的LLM时间
        result_list.append('failed')  # 超时也标记为失败

        # 使用收集到的最后一组反例作为最终赋值（如果有）
        if env_data['counterexamples_list'] and len(env_data['counterexamples_list']) > 0:
            last_assignments = env_data['counterexamples_list'][-1] if env_data['counterexamples_list'][-1] else []
            result_list.append(last_assignments)
        else:
            result_list.append([])

        # 使用收集到的完整反例列表
        result_list.append(env_data['counterexamples_list'])

        logger.info(f"超时处理完成: 保存了LLM时间={env_data['llm_time']:.3f}秒, 反例数量={len(env_data['counterexamples_list'])}")

        # 更新信息字典
        info_dict[file_path] = result_list
    else:
        # 从共享字典获取结果
        if 'result' in result_dict:
            info_dict[file_path] = result_dict['result']
            logger.info(f"文件 {file_path} 处理完成，结果已保存")
        else:
            logger.error(f"文件 {file_path} 处理失败，未返回结果")

    # 保存到文件，确保写入成功
    try:
        with open(args.info_dict_path, 'w') as file:
            json.dump(info_dict, file, indent=4)
            file.flush()  # 确保数据写入磁盘
        logger.info(f"成功保存信息到: {args.info_dict_path}")
    except Exception as e:
        logger.error(f"保存信息字典失败: {str(e)}")
        # 尝试备份保存
        try:
            backup_path = args.info_dict_path + ".backup"
            with open(backup_path, 'w') as file:
                json.dump(info_dict, file, indent=4)
            logger.info(f"已创建备份: {backup_path}")
        except Exception as e2:
            logger.error(f"创建备份也失败: {str(e2)}")

def _process_worker(file_path, list1, result_dict, env_data_queue, args):
    """
    实际处理文件的工作函数，在子进程中运行
    """
    try:
        # 在子进程中重新初始化日志记录器，使用相同的batch_id
        logger, _ = setup_logger(batch_id=args.batch_id)

        logger.info(f'子进程开始处理: {file_path}')

        # 设置子进程的设备
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            logger.info(f"子进程使用CUDA设备: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("子进程使用CPU")

        # 设置结果列表的默认值 [求解结果, 原始求解时间, 原始内存使用]
        result_list = [list1[0], list1[1], list1[2]]

        # 设置默认值，以防出现异常
        total_execution_time = 0
        total_solve_time = 0
        final_solve_time = 0
        llm_total_time = 0
        counterexamples_list = [[]]  # 默认反例列表

        # 清理内存，确保全新状态
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 读取SMT文件
        with open(file_path, 'r') as file:
            smtlib_str = file.read()
        dict_obj = json.loads(smtlib_str)
        smtlib_str = dict_obj['smt_script'] if 'smt-comp' in file_path else dict_obj['script']

        # 规范化SMT字符串
        smtlib_str, var_dict, constant_list = normalize_smt_str(smtlib_str)
        assertions = parse_smt2_string(smtlib_str)

        # 运行预测器
        start_time = time.time()
        env = None  # 确保env在外部作用域可见

        try:
            logger.info(f'开始执行: {file_path}')

            # 创建新的嵌入器实例
            embedder = CodeEmbedder_normalize()
            set_seed(0)  # 保持结果可复现

            # 初始化模型
            model = SimpleClassifier()
            model.load_state_dict(torch.load(args.binary_model_path))
            model.eval()

            model_time = EnhancedEightClassModel()
            model_time.load_state_dict(torch.load(args.eight_class_model_path))
            model_time.eval()

            # 使用断言的深拷贝，确保不共享状态
            z3_assertions = copy.deepcopy(assertions)

            # 创建环境，传入LLM主机参数
            env = ConstraintSimplificationEnv_test(
                embedder, z3_assertions, model, model_time, smtlib_str,
                file_path, var_dict, constant_list, args.solver,
                llm_host=args.llm_host, llm_model=args.llm_model
            )

            # 确认环境初始化状态
            logger.info(f"新环境创建完成: 变量数量={len(env.variables)}, LLM时间={env.llm_time:.3f}秒")
            if len(env.counterexamples_list) > 0:
                logger.info(f"初始反例列表: {env.counterexamples_list}")

            # 向主进程发送初始环境数据
            env_data_queue.put({
                'total_solve_time': env.total_solve_time,
                'final_solve_time': env.solve_time,
                'llm_time': env.llm_time,
                'counterexamples_list': env.counterexamples_list
            })

            # 设置和运行代理
            observation, action_space = env.reset()
            agent = create_agent(env, action_space)

            # 设置最大迭代次数
            max_iterations = args.num_episodes

            # 定义环境数据检查点
            last_check_time = time.time()
            check_interval = 5  # 每5秒检查一次

            # 定义回调函数，定期发送环境数据
            def data_callback(env, episode, step):
                nonlocal last_check_time
                current_time = time.time()
                if current_time - last_check_time >= check_interval:
                    # 向主进程发送更新的环境数据
                    env_data_queue.put({
                        'total_solve_time': env.total_solve_time,
                        'final_solve_time': env.solve_time,
                        'llm_time': env.llm_time,
                        'counterexamples_list': env.counterexamples_list
                    })
                    last_check_time = current_time
                    logger.debug(f"发送环境数据更新: LLM时间={env.llm_time:.3f}秒, 反例数量={len(env.counterexamples_list)}")
                return False  # 继续执行

            # 运行在线学习，但添加最大迭代次数限制和回调函数
            info = online_learning(
                agent=agent,
                env=env,
                number_of_episodes=max_iterations,
                print_every_x_episodes=1,
                record_period=args.record_period,
                callback=data_callback
            )

            # 获取环境中的数据
            total_solve_time = env.total_solve_time
            final_solve_time = env.solve_time
            llm_total_time = env.llm_time
            counterexamples_list = env.counterexamples_list

            # 最后再次发送更新的环境数据
            env_data_queue.put({
                'total_solve_time': total_solve_time,
                'final_solve_time': final_solve_time,
                'llm_time': llm_total_time,
                'counterexamples_list': counterexamples_list
            })

            logger.info(f'执行完成: {file_path}')
            logger.info(f'累计求解时间: {total_solve_time:.3f}秒')
            logger.info(f'最终求解时间: {final_solve_time:.3f}秒')
            logger.info(f'LLM累计时间: {llm_total_time:.3f}秒')

            # 检查并记录反例信息
            has_counterexamples = counterexamples_list and len(counterexamples_list) > 0
            counterexamples_count = len(counterexamples_list) if has_counterexamples else 0
            logger.info(f'反例集合数量: {counterexamples_count}')
            if has_counterexamples and len(counterexamples_list[-1]) > 0:
                logger.info(f'最后一组反例: {counterexamples_list[-1]}')

        except Exception as e:
            logger.error(f"执行过程中出现异常: {str(e)}")
            traceback.print_exc()

            # 如果环境已创建，在异常时也发送环境数据
            if env:
                try:
                    env_data_queue.put({
                        'total_solve_time': env.total_solve_time,
                        'final_solve_time': env.solve_time,
                        'llm_time': env.llm_time,
                        'counterexamples_list': env.counterexamples_list
                    })
                    logger.info(f"异常时发送环境数据: LLM时间={env.llm_time:.3f}秒")
                except Exception as e2:
                    logger.error(f"异常时发送环境数据失败: {str(e2)}")

        # 计算总执行时间
        end_time = time.time()
        total_execution_time = end_time - start_time

        # 更新结果列表，添加总执行时间、总求解时间、最终求解时间和LLM总时间
        # 确保只添加基本类型的值，不添加字典等复杂类型
        result_list.append(total_execution_time)
        result_list.append(total_solve_time)  # 添加求解器总时间
        result_list.append(final_solve_time)  # 添加最终成功求解时间
        result_list.append(llm_total_time)    # 添加LLM总时间

        # 添加求解状态，只使用'succeed'或'failed'
        try:
            result_list.append('succeed' if env and env.finish else 'failed')
        except (AttributeError, TypeError):
            result_list.append('failed')

        # 添加最终的赋值序列和所有反例集合
        has_counterexamples = counterexamples_list and len(counterexamples_list) > 0
        if has_counterexamples:
            # 记录最后一个赋值序列(如果有)
            last_assignments = counterexamples_list[-1] if counterexamples_list[-1] else []
            result_list.append(last_assignments)
            # 添加所有尝试过的反例集合
            result_list.append(counterexamples_list)
        else:
            result_list.append([])  # 没有最终赋值
            result_list.append([])  # 没有反例集合

        # 将结果保存到共享字典
        result_dict['result'] = result_list
        logger.info(f"子进程完成处理: {file_path}")

    except Exception as e:
        logger.error(f"子进程处理出错: {str(e)}")
        traceback.print_exc()
        # 确保即使出错也返回一个结果，使用'failed'状态
        result_dict['result'] = [list1[0], list1[1], list1[2], 0, 0, 0, 0, 'failed', [], [[]]]

def create_agent(env, action_space):
    """
    创建强化学习代理
    """
    return PearlAgent(
        policy_learner=SoftActorCritic(
            state_dim=768,
            action_space=action_space,
            actor_hidden_dims=[768, 512, 128],
            critic_hidden_dims=[768, 512, 128],
            action_representation_module=IdentityActionRepresentationModule(
                max_number_actions=action_space.n,
                representation_dim=action_space.action_dim,
            ),
        ),
        history_summarization_module=LSTMHistorySummarizationModule(
            observation_dim=768,
            action_dim=1,
            hidden_dim=768,
            history_length=len(env.variables),
        ),
        replay_buffer=FIFOOffPolicyReplayBuffer(10),
        device_id=-1,
    )

def main():
    parser = argparse.ArgumentParser(description='运行BVParti SMT约束求解预测器')

    # 添加命令行参数
    parser.add_argument('--rl_dict_path', type=str,
                        default='/home/lz/sibyl_3/src/networks/info_dict_rl.txt',
                        help='RL字典文件路径')
    parser.add_argument('--info_dict_path', type=str,
                        default='info_dict_SMTimer_llama3.1:70b_1200s_info_dict_rl_bvparti_0310.txt',
                        help='信息字典文件路径')
    parser.add_argument('--result_dict_path', type=str,
                        default='/home/lz/PycharmProjects/Pearl/test_rl/AriParti_sync/scripts/batch_output/bv_default/SMTimer_z3_result_rl.json',
                        help='结果字典文件路径')
    parser.add_argument('--binary_model_path', type=str,
                        default='/home/lz/PycharmProjects/Pearl/test_rl/test_cvc5/bvparti_process/models/binary_classifier.pth',
                        help='二分类模型路径')
    parser.add_argument('--eight_class_model_path', type=str,
                        default='/home/lz/PycharmProjects/Pearl/test_rl/test_cvc5/bvparti_process/models/eight_class_model.pth',
                        help='八分类模型路径')
    parser.add_argument('--time_threshold', type=int,
                        default=300,
                        help='时间阈值')
    parser.add_argument('--timeout', type=int,
                        default=1200,
                        help='执行超时时间（秒）')
    parser.add_argument('--num_episodes', type=int,
                        default=1,
                        help='训练轮数')
    parser.add_argument('--record_period', type=int,
                        default=1,
                        help='记录周期')
    parser.add_argument('--llm_host', type=str,
                        default='http://172.29.7.221:32793',
                        help='LLM服务器地址')
    parser.add_argument('--llm_model', type=str,
                        default='llama3.1:70b',
                        help='LLM模型名称')
    parser.add_argument('--solver', type=str,
                        default='bvparti',
                        choices=['z3', 'cvc5', 'bvparti'],
                        help='选择使用的求解器(z3/cvc5/bvparti)')

    args = parser.parse_args()
    run_predictor(args)

if __name__ == '__main__':
    main()
