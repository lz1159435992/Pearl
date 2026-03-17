"""
QF_NIA预测器运行模块
使用训练好的预测器模型和强化学习进行SMT求解实验
基于test_group_gai_6_llm_add_ce_predictor_SMTimer_docker_QF_NIA.py和run_predictor.py实现
"""
import os
import signal
import sys
import json
import random
import time
import copy
import traceback
import re
import math
import torch
import numpy as np
from tqdm import tqdm
from loguru import logger
from multiprocessing import Process, Manager, Queue, set_start_method, get_context

os.environ['ALL_PROXY'] = ''
os.environ['all_proxy'] = ''

# 添加项目路径
sys.path.append('/home/lz/PycharmProjects/Pearl')

from z3.z3 import parse_smt2_string, Solver, sat, unknown, unsat
from z3.z3 import Solver as Z3_Solver
from pearl.policy_learners.sequential_decision_making.soft_actor_critic import SoftActorCritic
from pearl.replay_buffers.sequential_decision_making.bootstrap_replay_buffer import FIFOOffPolicyReplayBuffer
from pearl.utils.functional_utils.experimentation.set_seed import set_seed
from pearl.action_representation_modules.identity_action_representation_module import IdentityActionRepresentationModule
from pearl.history_summarization_modules.lstm_history_summarization_module import LSTMHistorySummarizationModule
from pearl.pearl_agent import PearlAgent
from pearl.api.action_result import ActionResult
from pearl.api.environment import Environment
from pearl.utils.instantiations.spaces.discrete_action import DiscreteActionSpace

from test_rl.bert_embedder_test import CodeEmbedder_normalize
from test_rl.test_script.utils import (
    load_dictionary, normalize_smt_str, MyException, timeout_handler, setup_logger,
    find_var_declaration_in_string, split_at_check_sat, repalce_veriable, solve_assertion_get_range
)
from test_rl.test_script.online_learning_break import online_learning
from test_rl.test_QF_NIA.cvc5_process_QF_NIA.test_group_get_dis_smt_comp_bert_embeding_single import process_embeding
from test_rl.test_QF_NIA.cvc5_process_QF_NIA.train_predictor import (
    EnhancedClassifier,
    EnhancedEightClassModelLargeInput,
)
from ollama import Client
import tempfile
import subprocess

script_dir_local = os.path.dirname(os.path.abspath(__file__))
local_log_dir = os.path.join(script_dir_local, 'log')
os.makedirs(local_log_dir, exist_ok=True)
log_file_path = os.path.join(local_log_dir, 'run_predictor.log')
try:
    logger.remove()
except Exception:
    pass
logger.add(log_file_path, enqueue=True, rotation="100 MB", retention="14 days")


# 默认选择可用设备；在子进程中会再次设置为具体CUDA设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置多进程启动方法
try:
    set_start_method('spawn')
except RuntimeError:
    pass

def is_number(s):
    pattern = r'^(\d+|\d+\.\d+|\d+\/\d+)$'
    return re.match(pattern, s) is not None

def get_actions(tensor_1d_1):
    result_tensor = tensor_1d_1
    # 使用全局device（在worker内会被设置到CUDA）
    result_tensor = result_tensor.to(device)
    return result_tensor


class ConstraintSimplificationEnv_test(Environment):

    def __init__(self, embedder, z3ast, model, model_time, smtlib_str, file_path, var_dict, state,
                 solver_name='cvc5', llm_host=None, llm_model=None, global_timeout=None, time_dict=None):
        self.range_count = 10000
        self.var_dict = var_dict
        logger.info(self.var_dict)
        self.step_count = 0
        self.file_path = file_path
        self.actions_v = None
        self.embedder = embedder
        self.z3ast = z3ast
        self.z3ast_original = copy.deepcopy(z3ast)
        self.smtlib_str = smtlib_str
        self.smtlib_str_original = copy.deepcopy(smtlib_str)
        # LLM配置（必须从上层传入，默认由config.json提供）
        self.llm_host = llm_host
        self.llm_model = llm_model
        logger.info(f'环境初始化 - LLM配置: {llm_host} / {llm_model}')

        self.variables = sorted(list(self.var_dict.values()), key=lambda x: int(x.split('VAR')[1]))
        self.state_original = state
        self.state = None

        self.actions = []
        self.concrete_finish = False
        self.concrete_count = 0
        self.counterexamples_list = [[]]
        self.finish = False
        self.used_variables = []
        self.state_count = 0
        self.predictor = model
        self.predictor_time = model_time
        self.last_performance = 0
        self.solve_time = 0
        self.total_solve_time = 0
        self.llm_time = 0
        self.solver_name = solver_name
        self.solver = get_solver(solver_name)
        self.global_timeout = global_timeout
        self.v_related_assertions, self.var_range_dict = solve_assertion_get_range(self.z3ast, self.variables)
        self.var_range_dict_n = {}
        self.range_init()
        # QF_NIA时间类别映射：优先使用config.json提供的time_bins
        if time_dict and isinstance(time_dict, dict) and len(time_dict) > 0:
            try:
                self.time_dict = {int(k): int(v) for k, v in time_dict.items()}
            except Exception:
                self.time_dict = {0: 1, 1: 20, 2: 50, 3: 100, 4: 200, 5: 500, 6: 1200, 7: 20}
        else:
            self.time_dict = {0: 1, 1: 20, 2: 50, 3: 100, 4: 200, 5: 500, 6: 1200, 7: 20}

    def range_init(self):
        for variable in self.variables:
            type_info = find_var_declaration_in_string(self.smtlib_str_original, variable)
            if 'BitVec' in str(type_info):
                type_scale = str(type_info).split(' ')[-1]
                max_value = 2 ** int(type_scale) - 1
                min_value = 0
            elif type_info in ['Int', 'Real']:
                max_value = 2147483650
                min_value = -2147483650
                type_scale = 0
            else:
                # 兜底：未知类型按Int处理
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
        self.actions = get_actions(torch.arange(0, len(self.variables))).to(device)
        self.action_space = DiscreteActionSpace(self.actions)
        return self.state, self.action_space

    # 移除旧版基于Z3常量的处理函数，统一使用字符串结果版本（见下方 wrappers）

    def action_space(self):
        pass

    def process_text_python(self, text, variable_pred):
        variables = self.variables
        var_str = ','.join(variables)

        system_message = {
            "role": "system",
            "content": """ You are an advanced SAT/SMT solver, focusing on the optimization and resolution of logical constraint problems. 
            Your input consists of two parts: first, the counterexamples of failed solution assignments previously chosen, and second, the strings in SMT-LIB format that needs to be solved.
            You should analyze these inputs, using logical reasoning and heuristic methods to determine which variable assignments led to the failure of the solution,
            and identify the variable assignments that satisfy all constraint conditions. The output should be a specific value assignment for the target variable that can satisfy all the constraints defined in the strings.
            Your task is to find the specific values that should be assigned to the variables provided in the prompt to ensure that the entire constraint system is satisfiable.You should output only the numeric value,
            with an example as follows: <value> . Do not output any other text, explanations, or symbols.""" }
        user_message = {
            "role": "user",
            "content": text + f'This is the variable values from the previous failed SAT solving attempt and SMT text given to you in segments; analyze it. To speed up the solution and obtain a SAT result, provide a specific number that {variable_pred} should be assigned to. However, do not choose the values that have already failed to solve. Output only the numeric value. Do not output any other text, explanations, or symbols. The output must be a single number.'
        }

        llm_start_time = time.time()
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
            print(chunk['message']['content'], end='', flush=True)
        llm_elapsed = time.time() - llm_start_time
        self.llm_time += llm_elapsed
        logger.info(f"LLM调用耗时: {llm_elapsed:.3f}秒，累计LLM时间: {self.llm_time:.3f}秒")
        return responses

    def step(self, action):
        self.step_count += 1
        try:
            reward = 0
            action = self.action_space.actions_batch[action]
            action_v = action[0]
            variable_pred = self.variables[int(action_v.item())]

            # 初始化一组新的反例序列容器
            if self.concrete_count == 0:
                if len(self.counterexamples_list) > 0 and len(self.counterexamples_list[-1]) == 0:
                    pass
                else:
                    self.counterexamples_list.append([])

            ce_json = json.dumps(self.counterexamples_list)

            # 向LLM请求变量具体值
            text = "Here is the  counterexamples of failed solution assignments previously chosen in json formats:\n" + ce_json + "\n" \
                   + 'Here is the SMT file content:\n' + self.smtlib_str
            responses = self.process_text_python(text, variable_pred)
            index = len(responses) - 1
            while index > 0 and is_number(responses[index]) is False:
                index -= 1
            selected_int = responses[index]

            # 基本范围检查，错误则记录反例并reset
            if int(selected_int) < self.var_range_dict[variable_pred][0][0] or int(selected_int) > \
                    self.var_range_dict[variable_pred][0][1]:
                self.counterexamples_list[-1].append([variable_pred, selected_int])
                self.reset()

            type_info = find_var_declaration_in_string(self.smtlib_str_original, variable_pred)
            if 'BitVec' in str(type_info):
                type_scale = str(type_info).split(' ')[-1]
                new_constraint = "(assert (= {} (_ bv{} {})))\n".format(variable_pred, str(selected_int), type_scale)
            elif type_info in ['Int', 'Real']:
                new_constraint = "(assert (= {} {}))\n".format(variable_pred, str(selected_int))
                type_scale = 0
            else:
                new_constraint = "(assert (= {} {}))\n".format(variable_pred, str(selected_int))
                type_scale = 0

            related_assertions = self.v_related_assertions[variable_pred]
            count = 0
            if len(related_assertions) > 0:
                for a in related_assertions:
                    z3_solver_one = Z3_Solver()
                    z3_solver_one.add(a)
                    smtlib_str_before, smtlib_str_after = split_at_check_sat(z3_solver_one.to_smt2())
                    new_smtlib_str = smtlib_str_before + new_constraint + smtlib_str_after
                    solver_result = self.solver.solve(new_smtlib_str, timeout=10)
                    self.total_solve_time += solver_result.solve_time
                    if solver_result.result == 'sat':
                        count += 1
                        reward += 5

            logger.info(f"约束个数和通过的个数: {count}/{len(related_assertions)}")
            if count == len(related_assertions):
                if variable_pred not in self.used_variables:
                    self.used_variables.append(variable_pred)
                    self.concrete_count += 1
                    self.counterexamples_list[-1].append([variable_pred, selected_int])
                    smtlib_str_before, smtlib_str_after = split_at_check_sat(self.smtlib_str)
                    self.smtlib_str = smtlib_str_before + new_constraint + smtlib_str_after
                else:
                    for idx, value in enumerate(self.counterexamples_list[-1]):
                        if value[0] == variable_pred:
                            last_ce = copy.deepcopy(self.counterexamples_list[-1])
                            last_ce[idx] = [variable_pred, selected_int]
                            self.counterexamples_list.append(last_ce)
                    self.smtlib_str = repalce_veriable(self.smtlib_str, variable_pred, selected_int, type_scale, type_info)

                self.z3ast = parse_smt2_string(self.smtlib_str)
                reward += self.calculate_reward()
                self.state = process_embeding(self.smtlib_str, llm_host=self.llm_host, llm_model=self.llm_model).unsqueeze(0)
            else:
                return ActionResult(
                    observation=self.state,
                    reward=float(reward),
                    terminated=self.finish,
                    truncated=self.finish,
                    info={},
                    available_action_space=self.action_space,
                )

            del action
            del action_v
        except MyException:
            raise MyException("Timeout!")
        except Exception as e:
            logger.error(e)
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
            available_action_space=self.action_space,
        )

    @staticmethod
    def strings_to_onehot(string_list):
        str_to_index = {string: index for index, string in enumerate(string_list)}
        one_hot_tensors = []
        for string in string_list:
            one_hot_vector = torch.zeros(len(string_list), dtype=torch.float32)
            one_hot_vector[str_to_index[string]] = 1.0
            one_hot_vector.to(device)
            one_hot_tensors.append(one_hot_vector)
        one_hot_matrix = torch.stack(one_hot_tensors)
        del one_hot_tensors
        return one_hot_matrix

    @staticmethod
    def onehot_to_indices(one_hot_tensors):
        return torch.argmax(one_hot_tensors).item()

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

    def calculate_reward(self):
        performance = 0
        reward = 0
        count = 0
        if len(self.counterexamples_list) > 1:
            if self.counterexamples_list[-1] in self.counterexamples_list[:len(self.counterexamples_list) - 1]:
                reward += -10
                self.counterexamples_list.pop()
                return reward
            else:
                last_joined = ' '.join(' '.join(str(item) for item in inner_list) for inner_list in self.counterexamples_list[-1])
                for i in range(len(self.counterexamples_list) - 1):
                    current_joined = ' '.join(' '.join(str(item) for item in inner_list) for inner_list in self.counterexamples_list[i])
                    if last_joined in current_joined:
                        count += 1
                reward += self.counter_reward_function(len(self.counterexamples_list) - 1, len(self.counterexamples_list) - 1 - count)

        # 构建部分断言SMT字符串
        assertions = parse_smt2_string(self.smtlib_str)
        assertions_list = []
        for a in assertions:
            assertions_list.append(a)
        if len(assertions_list) > 0:
            indexes = random.sample(range(len(assertions_list)), max(1, int(len(assertions_list) * 0.5)))
        else:
            indexes = []
        z3_solver_part = Z3_Solver()
        for r in [assertions_list[i] for i in sorted(indexes)]:
            z3_solver_part.add(r)
        part_smt = z3_solver_part.to_smt2()
        new_state = process_embeding(part_smt, llm_host=self.llm_host, llm_model=self.llm_model).unsqueeze(0)
        new_state_dev = new_state.to(device)
        output = self.predictor(new_state_dev)
        predicted_solvability__part = (output > 0.5).int().item()
        if predicted_solvability__part == 1:
            reward += 5
            performance += 1
            output_time = self.predictor_time(new_state_dev)
            _, predicted_time = torch.max(output_time, 1)
            time_out = int(self.time_dict[int(predicted_time.item())] * 1000 * 1.2)
            part_timeout_sec = time_out/1000
            if self.global_timeout is not None:
                part_timeout_sec = min(part_timeout_sec, float(self.global_timeout))
            solver_result_part = self.solver.solve(part_smt, timeout=part_timeout_sec)
            self.total_solve_time += solver_result_part.solve_time
            reward, performance, finish = self.handle_case(solver_result_part.result, solver_result_part, time_out, reward, performance)

        # 全公式embedding用于最终预测
        new_state = process_embeding(self.smtlib_str, llm_host=self.llm_host, llm_model=self.llm_model).unsqueeze(0)
        new_state_dev = new_state.to(device)
        output = self.predictor(new_state_dev)
        predicted_solvability = (output > 0.5).int().item()
        if predicted_solvability == 1:
            reward += 5
            performance += 1
        logger.info(new_state.shape)
        logger.info(type(new_state))
        output_time = self.predictor_time(new_state_dev)
        _, predicted_time = torch.max(output_time, 1)
        time_out = int(self.time_dict[int(predicted_time.item())] * 1000 * 1.2)
        final_timeout_sec = time_out/1000
        if self.global_timeout is not None:
            final_timeout_sec = min(final_timeout_sec, float(self.global_timeout))
        solver_result = self.solver.solve(self.smtlib_str, timeout=final_timeout_sec)
        self.total_solve_time += solver_result.solve_time
        reward, performance, finish = self.handle_case(solver_result.result, solver_result, time_out, reward, performance)
        if finish:
            reward += int(1 / time_out * 500 * 1000)
            performance += 1
            self.finish = True
            self.solve_time = solver_result.solve_time
            logger.info(f"求解时间: {self.solve_time}")
        else:
            reward += -int(time_out / 10000)
        if performance < self.last_performance:
            self.reset()
        self.last_performance = performance
        return reward

    def handle_case(self, r, solver_result, time_out, reward, performance):
        if r == 'sat':
            return self.handle_satisfiable_wrapper(solver_result, time_out, reward, performance)
        elif r == 'unknown' or r == 'timeout':
            return self.handle_unknown_wrapper(solver_result, time_out, reward, performance)
        else:
            return self.handle_unsatisfiable_wrapper(time_out, reward, performance, solver_result)

    def handle_satisfiable_wrapper(self, solver_result, time_out, reward, performance):
        reward += int(1 / time_out * 500 * 1000)
        performance += 1
        logger.info(f"求解结果: sat, 本次求解耗时: {solver_result.solve_time:.3f}秒")
        return reward, performance, True

    def handle_unknown_wrapper(self, solver_result, time_out, reward, performance):
        reward += -int(time_out / 10000) / 2
        logger.info(f"求解结果: unknown, 本次求解耗时: {solver_result.solve_time:.3f}秒")
        return reward, performance, False

    def handle_unsatisfiable_wrapper(self, time_out, reward, performance, solver_result=None):
        reward += -int(time_out / 10000)
        if solver_result is not None:
            logger.info(f"求解结果: unsat, 本次求解耗时: {solver_result.solve_time:.3f}秒")
        return reward, performance, False


# 求解器封装
class SolverResult:
    def __init__(self, solve_time, result, model):
        self.solve_time = solve_time
        self.result = result
        self.model = model

class BaseSolver:
    def solve(self, smtlib_str, timeout=5):
        raise NotImplementedError

class Z3Solver(BaseSolver):
    def solve(self, smtlib_str, timeout=5):
        start = time.time()
        try:
            s = Z3_Solver()
            s.set("timeout", int(timeout * 1000))
            try:
                z3_exprs = parse_smt2_string(smtlib_str)
                if isinstance(z3_exprs, list):
                    s.add(*z3_exprs)
                else:
                    s.add(z3_exprs)
            except Exception as e:
                elapsed = time.time() - start
                return SolverResult(elapsed, 'parse_error', str(e))
            check_result = s.check()
            elapsed = time.time() - start
            if check_result == sat:
                return SolverResult(elapsed, 'sat', s.model().sexpr())
            elif check_result == unsat:
                return SolverResult(elapsed, 'unsat', None)
            elif check_result == unknown:
                return SolverResult(elapsed, 'unknown', s.reason_unknown())
            else:
                return SolverResult(elapsed, str(check_result), None)
        except Exception as e:
            elapsed = time.time() - start
            return SolverResult(elapsed, 'error', str(e))

class CVC5Solver(BaseSolver):
    def solve(self, smtlib_str, timeout=5):
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.smt2', delete=False) as f:
            f.write(smtlib_str)
            f.flush()
            cmd = ['cvc5', '--lang', 'smt2', '--produce-models', f.name, f'--tlimit={int(timeout*1000)}']
            start = time.time()
            try:
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
                    os.unlink(f.name)
                except:
                    pass
        return SolverResult(elapsed, result, model)

class MathSAT5Solver(BaseSolver):
    def solve(self, smtlib_str, timeout=5):
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.smt2', delete=False) as f:
            f.write(f"(set-option :timeout {int(timeout*1000)})\n")
            f.write(smtlib_str)
            f.flush()
            cmd = ['mathsat', f.name]
            start = time.time()
            try:
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
                    os.unlink(f.name)
                except:
                    pass
        return SolverResult(elapsed, result, model)

def get_solver(solver_name):
    solver_map = {
        'z3': Z3Solver(),
        'cvc5': CVC5Solver(),
        'mathsat': MathSAT5Solver(),
    }
    return solver_map.get(solver_name.lower(), CVC5Solver())


def create_agent(env, action_space):
    return PearlAgent(
        policy_learner=SoftActorCritic(
            state_dim=8192,
            action_space=action_space,
            actor_hidden_dims=[1024, 512, 128],
            critic_hidden_dims=[1024, 512, 128],
            action_representation_module=IdentityActionRepresentationModule(
                max_number_actions=action_space.n,
                representation_dim=action_space.action_dim,
            ),
        ),
        history_summarization_module=LSTMHistorySummarizationModule(
            observation_dim=8192,
            action_dim=1,
            hidden_dim=8192,
        ),
        replay_buffer=FIFOOffPolicyReplayBuffer(10),
        device_id=-1,
    )


def _process_worker(file_path, list1, embedding_path, shared_result, env_data_queue,
                    solver_name, llm_host, llm_model, models_dir, script_dir, timeout, time_bins):
    try:
        # 在子进程内设置设备，遵循参考实现：先设置CUDA设备，再进行任何CUDA相关操作
        global device
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        result_list = [list1[0], list1[1], list1[2]]
        total_solve_time = 0
        final_solve_time = 0
        llm_total_time = 0
        counterexamples_list = [[]]

        # 读取SMT文件
        with open(file_path, 'r') as file:
            smtlib_str = file.read()
        # 归一化
        smtlib_str, var_dict, constant_list = normalize_smt_str(smtlib_str)
        assertions = parse_smt2_string(smtlib_str)

        # 加载模型
        model = EnhancedClassifier(input_dim=8192)
        model_path = os.path.join(script_dir, models_dir, 'QF_NIA_bert_predictor_mask_best.pth')
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.to(device)
        model.eval()

        model_time = EnhancedEightClassModelLargeInput(input_dim=8192)
        model_time_path = os.path.join(script_dir, models_dir, 'QF_NIA_bert_predictor_2_mask_best_model.pth')
        model_time.load_state_dict(torch.load(model_time_path, map_location=torch.device('cpu')))
        model_time.to(device)
        model_time.eval()

        # 初始状态
        state = torch.tensor(np.load(embedding_path))

        # 创建环境（QF_NIA路径不使用本地 embedder，这里传 None 以减少内存占用）
        embedder = None
        set_seed(0)
        z3_assertions = copy.deepcopy(assertions)
        env = ConstraintSimplificationEnv_test(
            embedder, z3_assertions, model, model_time, smtlib_str,
            file_path, var_dict, state,
            solver_name=solver_name, llm_host=llm_host, llm_model=llm_model, global_timeout=timeout,
            time_dict=time_bins
        )

        # 初始环境数据发送
        env_data_queue.put({
            'total_solve_time': env.total_solve_time,
            'final_solve_time': env.solve_time,
            'llm_time': env.llm_time,
            'counterexamples_list': env.counterexamples_list,
        })

        # 运行代理
        observation, action_space = env.reset()
        agent = create_agent(env, action_space)

        # 回调以周期更新环境数据
        last_check_time = time.time()
        check_interval = 5
        def data_callback(env, episode, step):
            nonlocal last_check_time
            now = time.time()
            if now - last_check_time >= check_interval:
                env_data_queue.put({
                    'total_solve_time': env.total_solve_time,
                    'final_solve_time': env.solve_time,
                    'llm_time': env.llm_time,
                    'counterexamples_list': env.counterexamples_list,
                })
                last_check_time = now
            return False

        start_time = time.time()
        info = online_learning(
            agent=agent,
            env=env,
            number_of_episodes=1,
            print_every_x_episodes=1,
            record_period=1,
            callback=data_callback
        )

        total_solve_time = env.total_solve_time
        final_solve_time = env.solve_time
        llm_total_time = env.llm_time
        counterexamples_list = env.counterexamples_list

        end_time = time.time()
        total_execution_time = end_time - start_time

        # 组装结果
        result_list.append(total_execution_time)
        result_list.append(total_solve_time)
        result_list.append(final_solve_time)
        result_list.append(llm_total_time)
        status = 'succeed' if getattr(env, 'finish', False) else 'failed'
        result_list.append(status)
        if counterexamples_list and len(counterexamples_list) > 0:
            last_assignments = counterexamples_list[-1] if counterexamples_list[-1] else []
        else:
            last_assignments = []
        result_list.append(last_assignments)
        result_list.append(counterexamples_list if counterexamples_list else [])

        shared_result['result'] = result_list
    except Exception as e:
        # 异常日志写入（不影响结果结构）
        try:
            log_dir = os.path.join(script_dir, 'log')
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, 'worker_errors.log')
            with open(log_path, 'a', encoding='utf-8') as logf:
                logf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] file={file_path} solver={solver_name} host={llm_host} model={llm_model}\n")
                logf.write(f"embedding_path={embedding_path} models_dir={models_dir}\n")
                logf.write(f"Exception: {repr(e)}\n")
                logf.write(traceback.format_exc())
                logf.write('\n' + ('-'*80) + '\n')
        except Exception:
            pass
        # 失败回退
        shared_result['result'] = [list1[0], list1[1], list1[2], 0, 0, 0, 0, 'failed', [], [[]], 'worker_exception']


def process_single_file_with_timeout(file_path, list1, embedding_path, info_dict, info_name,
                                     solver_name, llm_host, llm_model, timeout, models_dir, script_dir, time_bins):
    ctx = get_context('spawn')
    manager = ctx.Manager()
    shared_result = manager.dict()
    env_data_queue = ctx.Queue()

    p = ctx.Process(target=_process_worker, args=(
        file_path, list1, embedding_path, shared_result, env_data_queue,
        solver_name, llm_host, llm_model, models_dir, script_dir, timeout, time_bins
    ))
    p.start()

    # 默认环境数据
    env_data = {
        'total_solve_time': 0,
        'final_solve_time': 0,
        'llm_time': 0,
        'counterexamples_list': [[]],
        'start_time': time.time(),
    }

    elapsed = 0
    check_interval = 1
    while elapsed < timeout:
        if not p.is_alive():
            break
        # 非阻塞获取更新
        try:
            while not env_data_queue.empty():
                new_env_data = env_data_queue.get_nowait()
                start_time = env_data['start_time']
                env_data.update(new_env_data)
                env_data['start_time'] = start_time
        except Exception:
            pass
        time.sleep(check_interval)
        elapsed = time.time() - env_data['start_time']

    if p.is_alive():
        try:
            while not env_data_queue.empty():
                new_env_data = env_data_queue.get_nowait()
                start_time = env_data['start_time']
                env_data.update(new_env_data)
                env_data['start_time'] = start_time
        except Exception:
            pass
        p.terminate()
        p.join()
        # 超时结果
        result_list = [list1[0], list1[1], list1[2]]
        result_list.append(timeout)
        result_list.append(env_data['total_solve_time'])
        result_list.append(env_data['final_solve_time'])
        result_list.append(env_data['llm_time'])
        result_list.append('failed')
        if env_data['counterexamples_list'] and len(env_data['counterexamples_list']) > 0:
            last_assignments = env_data['counterexamples_list'][-1] if env_data['counterexamples_list'][-1] else []
            result_list.append(last_assignments)
        else:
            result_list.append([])
        result_list.append(env_data['counterexamples_list'])
        result_list.append('worker_timeout')
    else:
        if 'result' in shared_result:
            result_list = shared_result['result']
        else:
            result_list = [list1[0], list1[1], list1[2], 0, 0, 0, 0, 'failed', [], [[]], 'worker_no_result']

    # 写入并返回状态
    info_dict[file_path] = result_list
    try:
        tmp_path = info_name + '.tmp'
        with open(tmp_path, 'w') as file:
            json.dump(info_dict, file, indent=4)
        os.replace(tmp_path, info_name)
    except Exception as e:
        logger.error(f'写入结果文件失败: {e}')

    status = result_list[7]
    return result_list, status

def load_config():
    """
    加载配置文件
    
    Returns:
        config: 配置字典
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.json')
    
    if not os.path.exists(config_path):
        logger.warning(f'配置文件不存在: {config_path}，使用默认配置')
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    logger.info(f'已加载配置文件: {config_path}')
    return config


def run_QF_NIA_experiment(
    solver_name=None,
    llm_host=None,
    llm_model=None,
    timeout=None,
    max_files=None,
    config=None
):
    """
    运行QF_NIA求解实验
    
    Args:
        solver_name: 使用的SMT求解器名称 (cvc5, z3, mathsat, etc.)
        llm_host: LLM服务器地址
        llm_model: LLM模型名称
        timeout: 单个问题的超时时间(秒)
        max_files: 最多处理的文件数(None表示处理所有)
        config: 配置字典（从config.json加载）
    """
    # 如果没有提供config，加载默认配置
    if config is None:
        config = load_config()
    
    # 从配置文件中获取默认值（如果参数未指定）
    if solver_name is None:
        solver_name = config.get('solver', {}).get('name', 'cvc5')
    if llm_host is None:
        llm_host = config.get('embedding', {}).get('host', 'http://172.29.7.221:32903')
    if llm_model is None:
        llm_model = config.get('embedding', {}).get('model', 'llama3.1:70b')
    if timeout is None:
        timeout = config.get('solver', {}).get('timeout', 1200)
    
    logger.info('='*80)
    logger.info(f'开始QF_NIA求解实验')
    logger.info(f'求解器: {solver_name}')
    logger.info(f'LLM主机: {llm_host}')
    logger.info(f'LLM模型: {llm_model}')
    logger.info(f'超时时间: {timeout}秒')
    if max_files:
        logger.info(f'最大文件数: {max_files}')
    logger.info('='*80)
    
    # 读取QF_NIA问题和求解结果（从config.data.source读取）
    data_source = config.get('data', {}).get('source', '/home/lz/PycharmProjects/Pearl/test_rl/test_QF_NIA/cvc5_QF_NIA.json')
    with open(data_source, 'r') as file:
        solve_dict = json.load(file)
    try:
        for _k, _v in solve_dict.items():
            if isinstance(_v, list) and len(_v) > 0 and _v[0] == 'timeout':
                _v[0] = 'unknown'
    except Exception:
        pass
    logger.info(f'加载求解结果: {data_source} (共{len(solve_dict)}条)')
    
    # 设置结果保存文件
    info_name = f'info_dict_SMTimer_{solver_name}_{llm_model.replace(":", "_")}_QF_NIA.txt'
    info_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), info_name)
    if not os.path.exists(info_name):
        info_dict = {}
        with open(info_name, 'w') as file:
            json.dump(info_dict, file, indent=4)
        logger.info(f'创建新的结果文件: {info_name}')
    else:
        info_dict = load_dictionary(info_name)
        logger.info(f'加载已有结果文件: {info_name} (已有{len(info_dict)}条记录)')
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 从配置文件中获取路径
    paths_config = config.get('paths', {})
    test_json_name = paths_config.get('test_json', 'QF_NIA_test.json')
    models_dir = paths_config.get('models_dir', 'models')
    # 从配置文件获取时间bins（用于驱动环境内time_dict）
    time_bins = config.get('time_bins', {})
    
    # 读取测试集
    test_json_path = os.path.join(script_dir, test_json_name)
    with open(test_json_path, 'r') as file:
        result_dict = json.load(file)
    logger.info(f'加载测试集: {test_json_path} ({len(result_dict)} 个样本)')
    
    # 随机打乱顺序
    items = list(result_dict.items())
    random.shuffle(items)
    result_dict = dict(items)
    
    # 预测器模型由子进程加载，主进程不再重复加载
    
    # 统计信息
    total_processed = 0
    total_success = 0
    total_failed = 0
    total_skipped = 0
    
    # 计算总数和已处理数
    total_files = len(result_dict)
    already_processed = len(info_dict)
    
    # 预先过滤出需要处理的文件（排除已处理的，排除不符合基本条件的）
    # 同时统计所有符合条件的文件（包括已处理的和待处理的），用于设置进度条总数
    files_to_process = []
    eligible_files_count = 0  # 所有符合条件的文件数（包括已处理的和待处理的）
    already_processed_eligible = 0  # 已处理且符合条件的文件数
    
    for key, value in result_dict.items():
        # 检查是否为QF_NIA问题
        if 'QF_NIA' not in key:
            continue
        
        # 检查embedding文件是否存在
        if len(value) < 1 or not os.path.exists(value[0]):
            continue
        
        # 检查是否在solve_dict中
        if key not in solve_dict:
            continue
        
        list1 = solve_dict[key]
        
        # 只处理sat或unknown且求解时间>300s的问题
        if list1[0] not in ["sat", "unknown"]:
            continue
        
        if list1[1] <= 300:
            continue
        
        # 跳过特定无法处理的文件
        if '20220315-MathProblems' in key:
            continue
        
        # 符合所有基本条件，计入符合条件的文件总数
        eligible_files_count += 1
        
        # 检查是否已经处理过
        if key in info_dict.keys():
            already_processed_eligible += 1
            continue
        
        # 符合所有条件且未处理，加入待处理列表
        files_to_process.append((key, value))
    
    # 计算实际需要处理的文件数
    files_to_process_count = len(files_to_process)
    
    # 验证逻辑一致性：符合条件的文件数应该等于已处理数加上待处理数
    if eligible_files_count != already_processed_eligible + files_to_process_count:
        logger.warning(f'逻辑不一致: 符合条件的文件数({eligible_files_count}) != 已处理数({already_processed_eligible}) + 待处理数({files_to_process_count})')
        # 修正：使用实际计算的值
        eligible_files_count = already_processed_eligible + files_to_process_count
    
    # 创建进度条
    logger.info(f'开始处理测试集: 总计{total_files}个文件, 符合条件的{eligible_files_count}个(已处理{already_processed_eligible}个, 待处理{files_to_process_count}个)')
    
    # 遍历需要处理的文件（带进度条）
    # 进度条总数设置为符合条件的文件总数（包括已处理的和待处理的）
    # 初始值设置为已处理且符合条件的文件数，这样进度条从一开始就显示正确的进度
    # 注意：tqdm会自动为每个迭代的项目递增进度条，即使使用continue跳过也会递增
    # 因此进度条反映的是"尝试处理的文件数"，而不是"实际处理的文件数"
    pbar = tqdm(
        files_to_process, 
        total=eligible_files_count,
        initial=already_processed_eligible,
        desc=f'RL求解进度',
        unit='file',
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
    )
    
    for key, value in pbar:
        # 限制处理文件数
        if max_files is not None and total_processed >= max_files:
            logger.info(f'达到最大处理文件数限制: {max_files}')
            break
        
        # 注意: 以下检查已在预过滤中完成，但为了安全起见保留关键检查
        # 检查是否为QF_NIA问题（预过滤已处理，但保留作为安全检查）
        if 'QF_NIA' not in key:
            total_skipped += 1
            pbar.set_postfix({'处理': total_processed, '成功': total_success, '失败': total_failed, '跳过': total_skipped})
            continue
        
        # 检查embedding文件是否存在（预过滤已处理，但保留作为安全检查）
        if len(value) < 1 or not os.path.exists(value[0]):
            logger.warning(f'Embedding文件不存在: {key}')
            total_skipped += 1
            pbar.set_postfix({'处理': total_processed, '成功': total_success, '失败': total_failed, '跳过': total_skipped})
            continue
        
        # 初始embedding由子进程加载
        
        # 获取原始求解结果（预过滤已处理，但保留作为安全检查）
        if key not in solve_dict:
            logger.warning(f'未找到求解结果: {key}')
            total_skipped += 1
            pbar.set_postfix({'处理': total_processed, '成功': total_success, '失败': total_failed, '跳过': total_skipped})
            continue
        
        list1 = solve_dict[key]
        logger.info(f'处理文件: {key}, 状态: {list1}')
        
        # 只处理sat或unknown且求解时间>300s的问题（预过滤已处理，但保留作为安全检查）
        if list1[0] not in ["sat", "unknown"]:
            total_skipped += 1
            pbar.set_postfix({'处理': total_processed, '成功': total_success, '失败': total_failed, '跳过': total_skipped})
            continue
        
        if list1[1] <= 300:
            total_skipped += 1
            pbar.set_postfix({'处理': total_processed, '成功': total_success, '失败': total_failed, '跳过': total_skipped})
            continue
        
        file_path = key

        def _persist_result(_file_path, _result_list):
            info_dict[_file_path] = _result_list
            try:
                tmp_path = info_name + '.tmp'
                with open(tmp_path, 'w') as file:
                    json.dump(info_dict, file, indent=4)
                os.replace(tmp_path, info_name)
            except Exception as e:
                logger.error(f'写入结果文件失败: {e}')

        def _persist_failed(_file_path, _list1, _reason):
            _persist_result(_file_path, [_list1[0], _list1[1], _list1[2], 0, 0, 0, 0, 'failed', [], [[]], _reason])
        
        # 检查是否已经处理过（预过滤已处理，但保留作为安全检查，防止并发情况）
        if file_path in info_dict.keys():
            logger.info(f'文件已处理，跳过: {file_path}')
            total_skipped += 1
            pbar.set_postfix({'处理': total_processed, '成功': total_success, '失败': total_failed, '跳过': total_skipped})
            continue
        
        # 跳过特定无法处理的文件（预过滤已处理，但保留作为安全检查）
        if '20220315-MathProblems' in file_path:
            logger.info(f'跳过无法处理的文件: {file_path}')
            total_skipped += 1
            pbar.set_postfix({'处理': total_processed, '成功': total_success, '失败': total_failed, '跳过': total_skipped})
            continue
        
        # 读取SMT文件
        try:
            with open(file_path, 'r') as file:
                smtlib_str = file.read()
        except Exception as e:
            logger.error(f'读取文件失败: {file_path}, 错误: {e}')
            total_failed += 1
            _persist_failed(file_path, list1, 'read_file_error')
            pbar.set_postfix({'处理': total_processed, '成功': total_success, '失败': total_failed, '跳过': total_skipped})
            continue
        
        # 归一化SMT字符串
        signal.alarm(30)
        signal.signal(signal.SIGALRM, timeout_handler)
        try:
            smtlib_str, var_dict, constant_list = normalize_smt_str(smtlib_str)
        except MyException:
            logger.info(f'归一化超时: {file_path}')
            signal.alarm(0)
            total_failed += 1
            _persist_failed(file_path, list1, 'normalize_timeout')
            pbar.set_postfix({'处理': total_processed, '成功': total_success, '失败': total_failed, '跳过': total_skipped})
            continue
        except Exception as e:
            logger.error(f'归一化失败: {file_path}, 错误: {e}')
            signal.alarm(0)
            total_failed += 1
            _persist_failed(file_path, list1, 'normalize_error')
            pbar.set_postfix({'处理': total_processed, '成功': total_success, '失败': total_failed, '跳过': total_skipped})
            continue
        signal.alarm(0)
        
        if var_dict is None:
            logger.warning(f'归一化返回空变量字典: {file_path}')
            total_failed += 1
            _persist_failed(file_path, list1, 'var_dict_none')
            pbar.set_postfix({'处理': total_processed, '成功': total_success, '失败': total_failed, '跳过': total_skipped})
            continue
        
        # 解析由子进程完成，这里不再提前解析
        
        # 采用多进程worker处理单个文件
        try:
            embedding_path = value[0]
            result_list, status = process_single_file_with_timeout(
                file_path, list1, embedding_path, info_dict, info_name,
                solver_name, llm_host, llm_model, timeout, models_dir, script_dir, time_bins
            )
            if status == 'succeed':
                total_success += 1
            else:
                total_failed += 1
        except Exception as e:
            logger.error(f'多进程处理失败: {e}')
            total_failed += 1
            _persist_failed(file_path, list1, 'worker_exception')
        
        total_processed += 1
        
        # 更新进度条信息
        pbar.set_postfix({
            '处理': total_processed,
            '成功': total_success,
            '失败': total_failed,
            '跳过': total_skipped
        })
    
    # 关闭进度条
    pbar.close()
    
    # 输出最终统计
    logger.info('='*80)
    logger.info('实验完成！')
    logger.info(f'总处理: {total_processed}')
    logger.info(f'成功: {total_success}')
    logger.info(f'失败: {total_failed}')
    logger.info(f'跳过: {total_skipped}')
    logger.info(f'结果已保存到: {info_name}')
    logger.info('='*80)


if __name__ == '__main__':
    import argparse
    
    # 首先加载配置文件
    config = load_config()
    
    # 从配置文件中获取默认值
    default_solver = config.get('solver', {}).get('name', 'cvc5')
    default_llm_host = config.get('embedding', {}).get('host', 'http://172.29.7.221:32903')
    default_llm_model = config.get('embedding', {}).get('model', 'llama3.1:70b')
    default_timeout = config.get('solver', {}).get('timeout', 1200)
    
    # 创建参数解析器，使用配置文件中的默认值
    parser = argparse.ArgumentParser(
        description='运行QF_NIA求解实验',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'''
配置文件: config.json
当前默认值:
  求解器: {default_solver}
  LLM主机: {default_llm_host}
  LLM模型: {default_llm_model}
  超时时间: {default_timeout}秒

命令行参数会覆盖配置文件中的值。
        '''
    )
    
    parser.add_argument('--solver', type=str, default=None, 
                        help=f'SMT求解器名称 (默认: {default_solver})')
    parser.add_argument('--llm_host', type=str, default=None, 
                        help=f'LLM服务器地址 (默认: {default_llm_host})')
    parser.add_argument('--llm_model', type=str, default=None, 
                        help=f'LLM模型名称 (默认: {default_llm_model})')
    parser.add_argument('--timeout', type=int, default=None, 
                        help=f'超时时间(秒) (默认: {default_timeout})')
    parser.add_argument('--max_files', type=int, default=None, 
                        help='最大处理文件数 (默认: 无限制)')
    
    args = parser.parse_args()
    
    # 运行实验，传入config和命令行参数
    # 命令行参数会覆盖config中的默认值
    run_QF_NIA_experiment(
        solver_name=args.solver,
        llm_host=args.llm_host,
        llm_model=args.llm_model,
        timeout=args.timeout,
        max_files=args.max_files,
        config=config
    )

