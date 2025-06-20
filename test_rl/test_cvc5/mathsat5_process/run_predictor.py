import os
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
from decimal import Decimal, getcontext
from loguru import logger
import torch
import numpy as np
from z3 import *
from z3.z3 import parse_smt2_string, Solver, sat, unknown, unsat
from openai import OpenAI
from ollama import Client

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

from test_rl.test_script.utils import (
    parse_smt2_in_parts, process_smt_lib_string, fetch_data_as_dict,
    solve_and_measure_time, model_to_dict, load_dictionary, extract_variables_from_smt2_content,
    normalize_variables, normalize_smt_str, MyException, timeout_handler, setup_logger,
    find_var_declaration_in_string, split_at_check_sat, find_assertions_related_to_var_name,
    find_assertions_related_to_var_names_optimized, repalce_veriable, normalize_smt_str_without_replace,
    solve_assertion_get_range
)

from test_rl.bert_embedder_test import CodeEmbedder_normalize
from test_rl.bert_predictor_2_mask import EnhancedEightClassModel
from test_rl.bert_predictor_mask import SimpleClassifier
from test_rl.test_script.online_learning_break import online_learning

# 设置环境变量
os.environ['ALL_PROXY'] = ''
os.environ['all_proxy'] = ''

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

class ConstraintSimplificationEnv_test(Environment):
    def __init__(self, embedder, z3ast, model, model_time, smtlib_str, file_path, var_dict, constant_list, solver_name='z3', llm_host='http://172.29.7.221:32903', llm_model='llama3.1:70b'):
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
        self.state_count = 0
        self.predictor = model
        self.predictor_time = model_time
        self.last_performance = 0
        self.solve_time = 0  # 总求解时间
        self.total_solve_time = 0  # 累计所有求解操作的时间
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
        
        self.actions = get_actions(torch.arange(0, len(self.variables)))
        self.actions.to(device)
        self.action_space = DiscreteActionSpace(self.actions)
        del self.actions
        torch.cuda.empty_cache()
        return self.state, self.action_space

    def handle_case(self, r, solver_result, time_out, reward, performance):
        if r == sat:
            reward += int(1 / time_out * 500 * 1000)
            performance += 1
            self.total_solve_time += solver_result.solve_time  # 累加求解时间
            return reward, performance, True
        elif r == unknown:
            reward += -int(time_out / 10000) / 2
            self.total_solve_time += solver_result.solve_time  # 累加求解时间
            return reward, performance, False
        else:
            reward += -int(time_out / 10000)
            self.total_solve_time += solver_result.solve_time  # 累加求解时间
            return reward, performance, False


    def process_text_python(self, text, variable_pred):
        variables = self.variables
        var_str = ','.join(variables)

        system_message = {
            "role": "system",
            "content": """You are an advanced SAT/SMT solver, focusing on the optimization and resolution of logical constraint problems. 
            Your input consists of two parts: first, the counterexamples of failed solution assignments previously chosen, and second, the strings in SMT-LIB format that needs to be solved.
            You should analyze these inputs, using logical reasoning and heuristic methods to determine which variable assignments led to the failure of the solution,
            and identify the variable assignments that satisfy all constraint conditions. The output should be a set of specific variable assignments that can satisfy all the constraints defined in the strings.
            Your task is to find the specific values that should be assigned to the variables provided in the prompt to ensure that the entire constraint system is satisfiable.You should output only the numeric value,
            with an example as follows: <value> . Do not output any other text, explanations, or symbols."""}
        user_message = {
            "role": "user",
            "content": text + f'This is the The variable values from the previous failed SAT solving attempt and SMT text given to you in segments; analyze it. To speed up the solution and obtain a SAT result, provide a specific number that {variable_pred} should be assigned to. However, do not choose the values that have already failed to solve. Output only the numeric value. Do not output any other text, explanations, or symbols. The output must be a single number.'
        }

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
                    # 构建包含新约束的SMT字符串
                    solver_str = str(a)
                    smtlib_str_before, smtlib_str_after = split_at_check_sat(solver_str)
                    new_smtlib_str = smtlib_str_before + new_constraint + smtlib_str_after
                    
                    # 使用配置的求解器进行求解
                    solver_result = self.solver.solve(new_smtlib_str, timeout=10)
                    if solver_result.result == 'sat':
                        count += 1
                        reward += 5
                    self.total_solve_time += solver_result.solve_time

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

                # 使用配置的求解器进行求解
                solver_result = self.solver.solve(self.smtlib_str, timeout=10)
                reward += self.calculate_reward(solver_result)
                
                # 更新状态
                self.z3ast = parse_smt2_string(self.smtlib_str)
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

    def calculate_reward(self, solver_result):
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

        # 使用较短的超时时间进行初步检查
        solver_result = self.solver.solve(self.smtlib_str, timeout=10)  # 10秒
        output = self.predictor(self.state)
        predicted_solvability__part = (output > 0.5).int().item()
        if predicted_solvability__part == 1:
            reward += 5
            performance += 1

            output_time = self.predictor_time(self.state)
            _, predicted_time = torch.max(output_time, 1)
            time_out = self.time_dict[int(predicted_time.item())]  # 时间字典中的值已经是秒为单位

            # 使用预测的时间进行求解
            solver_result = self.solver.solve(self.smtlib_str, timeout=time_out)  # 直接使用秒为单位
            reward, performance, finish = self.handle_case(solver_result.result, solver_result, time_out*1000, reward, performance)

        output = self.predictor(self.state)
        predicted_solvability = (output > 0.5).int().item()
        if predicted_solvability == 1:
            reward += 5
            performance += 1

        output_time = self.predictor_time(self.state)
        _, predicted_time = torch.max(output_time, 1)
        time_out = self.time_dict[int(predicted_time.item())]  # 时间字典中的值已经是秒为单位

        # 使用预测的时间进行最终求解
        solver_result = self.solver.solve(self.smtlib_str, timeout=time_out)  # 直接使用秒为单位
        reward, performance, finish = self.handle_case(solver_result.result, solver_result, time_out*1000, reward, performance)
        if finish:
            reward += int(1 / (time_out*1000) * 500 * 1000)
            performance += 1
            self.finish = True
            # 记录最终成功求解的时间，而不是总求解时间
            self.solve_time = solver_result.solve_time
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

# 添加求解器基类和实现
class SolverResult:
    def __init__(self, solve_time, result, model):
        self.solve_time = solve_time
        self.result = result
        self.model = model

class Solver:
    def solve(self, smtlib_str, timeout=5):
        raise NotImplementedError

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

class MathSAT5Solver(Solver):
    def solve(self, smtlib_str, timeout=5):
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.smt2', delete=False) as f:
            f.write(f"(set-option :timeout {int(timeout*1000)})\n")  # MathSAT使用毫秒为单位
            f.write(smtlib_str)
            f.flush()
            cmd = ['mathsat', f.name]
            logger.info(f"Running command: {' '.join(cmd)}")
            start = time.time()
            try:
                # 进程超时设置比求解器超时稍长一点
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
        'mathsat': MathSAT5Solver(),
    }
    return solver_map.get(solver_name.lower(), Z3Solver())  # 默认使用Z3求解器

def run_predictor(args):
    """
    运行预测器的主函数
    """
    setup_logger()

    # 加载RL字典
    with open(args.rl_dict_path, 'r') as file:
        rl_dict = json.load(file)

    # 创建或加载信息字典
    if not os.path.exists(args.info_dict_path):
        info_dict = {}
        with open(args.info_dict_path, 'w') as file:
            json.dump(info_dict, file, indent=4)
        logger.info(f'文件{args.info_dict_path} 已创建。')
    else:
        info_dict = load_dictionary(args.info_dict_path)
        logger.info('文件已存在。')

    # 加载结果字典
    with open(args.result_dict_path, 'r') as file:
        result_dict = json.load(file)

    # 处理每个结果
    for key, value in result_dict.items():
        list1 = value
        if list1[0] in ["sat", "unknown"] and list1[1] > args.time_threshold and key in rl_dict.keys():
            logger.info(f'处理文件: {key}')
            process_single_file(key, list1, info_dict, args)

def process_single_file(file_path, list1, info_dict, args):
    """
    处理单个文件的函数
    """
    if file_path in info_dict.keys() or 'gnu_angr.tar.gz/single_test/cat/cat43772' in file_path:
        return

    try:
        # 读取和处理SMT文件
        with open(file_path, 'r') as file:
            smtlib_str = file.read()
        dict_obj = json.loads(smtlib_str)
        smtlib_str = dict_obj['smt_script'] if 'smt-comp' in file_path else dict_obj['script']
        
        # 规范化SMT字符串
        smtlib_str, var_dict, constant_list = normalize_smt_str(smtlib_str)
        assertions = parse_smt2_string(smtlib_str)
        solver = Solver()
        for a in assertions:
            solver.add(a)

        # 设置结果列表 [求解结果, 原始求解时间, 原始内存使用]
        result_list = [list1[0], list1[1], list1[2]]

        # 创建环境实例，用于在超时情况下访问
        env = None
        
        # 运行预测器
        start_time = time.time()
        signal.alarm(args.timeout)
        signal.signal(signal.SIGALRM, timeout_handler)

        try:
            logger.info(f'开始执行: {file_path}')
            env = run_predictor_for_file(file_path, assertions, smtlib_str, var_dict, constant_list, args)
            # 获取求解器累计的求解时间和最终成功求解时间
            total_solve_time = env.total_solve_time if hasattr(env, 'total_solve_time') else 0
            final_solve_time = env.solve_time if hasattr(env, 'solve_time') else 0
        except MyException:
            logger.info(f'执行超时: {file_path}')
            # 计算已经过去的时间作为总执行时间
            end_time = time.time()
            total_execution_time = end_time - start_time
            
            # 在超时情况下，尝试从环境中获取已累计的求解器时间
            total_solve_time = env.total_solve_time if env and hasattr(env, 'total_solve_time') else 0
            final_solve_time = env.solve_time if env and hasattr(env, 'solve_time') else 0
            logger.info(f'超时时记录的总求解器时间: {total_solve_time}秒')
            logger.info(f'超时时记录的最终求解时间: {final_solve_time}秒')
            
            # 更新结果列表，添加总执行时间、总求解时间和最终求解时间
            result_list.append(total_execution_time)
            result_list.append(total_solve_time)
            result_list.append(final_solve_time)
            
            # 更新信息字典并返回
            update_info_dict(info_dict, file_path, result_list, args)
            signal.alarm(0)
            return

        signal.alarm(0)
        end_time = time.time()
        total_execution_time = end_time - start_time
        
        # 更新结果列表，添加总执行时间、总求解时间和最终求解时间
        result_list.append(total_execution_time)
        result_list.append(total_solve_time)  # 添加求解器总时间
        result_list.append(final_solve_time)  # 添加最终成功求解时间

        # 更新信息字典
        update_info_dict(info_dict, file_path, result_list, args)

    except Exception as e:
        logger.error(f'处理文件 {file_path} 时出错: {str(e)}')
        traceback.print_exc()

def run_predictor_for_file(file_path, assertions, smtlib_str, var_dict, constant_list, args):
    """
    为单个文件运行预测器
    
    Args:
        file_path: 文件路径
        assertions: 断言列表
        smtlib_str: SMT-LIB格式字符串
        var_dict: 变量字典
        constant_list: 常量列表
        args: 命令行参数
    
    Returns:
        env: 环境实例
    """
    embedder = CodeEmbedder_normalize()
    set_seed(0)

    # 初始化模型
    model = SimpleClassifier()
    model.load_state_dict(torch.load(args.binary_model_path))
    model.eval()

    model_time = EnhancedEightClassModel()
    model_time.load_state_dict(torch.load(args.eight_class_model_path))
    model_time.eval()

    # 创建环境，传入LLM主机参数
    env = ConstraintSimplificationEnv_test(
        embedder, assertions, model, model_time, smtlib_str,
        file_path, var_dict, constant_list, args.solver, 
        llm_host=args.llm_host, llm_model=args.llm_model
    )

    # 设置和运行代理
    observation, action_space = env.reset()
    agent = create_agent(env, action_space)
    
    info = online_learning(
        agent=agent,
        env=env,
        number_of_episodes=args.num_episodes,
        print_every_x_episodes=1,
        record_period=args.record_period,
    )

    return env

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

def update_info_dict(info_dict, file_path, result_list, args):
    """
    更新并保存信息字典
    """
    info_dict[file_path] = result_list
    with open(args.info_dict_path, 'w') as file:
        json.dump(info_dict, file, indent=4)

def main():
    parser = argparse.ArgumentParser(description='运行SMT约束求解预测器')
    
    # 添加命令行参数
    parser.add_argument('--rl_dict_path', type=str, 
                        default='/home/lz/sibyl_3/src/networks/info_dict_rl.txt',
                        help='RL字典文件路径')
    parser.add_argument('--info_dict_path', type=str, 
                        default='info_dict_gai_6_normal_0107_pre_SMTimer_deepseek-r1:70b_1200s_info_dict_rl.txt',
                        help='信息字典文件路径')
    parser.add_argument('--result_dict_path', type=str, 
                        default='/home/lz/PycharmProjects/Pearl/test_rl/test_solve/info_dict_bingxing.txt',
                        help='结果字典文件路径')
    parser.add_argument('--binary_model_path', type=str,
                        default='bert_predictor_mask_best.pth',
                        help='二分类模型路径')
    parser.add_argument('--eight_class_model_path', type=str,
                        default='bert_predictor_2_mask_best_model.pth',
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
                        default='http://172.29.7.221:32903',
                        help='LLM服务器地址')
    parser.add_argument('--llm_model', type=str,
                        default='llama3.1:70b',
                        help='LLM模型名称')
    parser.add_argument('--solver', type=str,
                        default='z3',
                        choices=['z3', 'cvc5', 'mathsat'],
                        help='选择使用的求解器(z3/cvc5/mathsat)')

    args = parser.parse_args()
    run_predictor(args)

if __name__ == '__main__':
    main() 