import os
import signal
import json
import random
import time
import torch
import numpy as np
from z3.z3 import parse_smt2_string, Solver
from multiprocessing import Process, Queue
from loguru import logger

from bert_embedder_test import CodeEmbedder_normalize
from test_rl.predictor.smt_comp_NIA.bert_predictor_mask_llm import EnhancedClassifier
from test_rl.predictor.smt_comp_NIA.bert_predictor_2_mask_llm import EnhancedEightClassModelLargeInput
from test_rl.test_script.utils import (
    load_dictionary, normalize_smt_str, MyException, timeout_handler
)
from test_rl.test_script.online_learning_break import online_learning
from env_gai_6_llm_add_ce_predictor_docker_llm_embed import ConstraintSimplificationEnv_test
from pearl.pearl_agent import PearlAgent
from pearl.policy_learners.sequential_decision_making.soft_actor_critic import SoftActorCritic
from pearl.replay_buffers.sequential_decision_making.bootstrap_replay_buffer import FIFOOffPolicyReplayBuffer
from pearl.utils.functional_utils.experimentation.set_seed import set_seed
from pearl.action_representation_modules.identity_action_representation_module import IdentityActionRepresentationModule
from pearl.history_summarization_modules.lstm_history_summarization_module import LSTMHistorySummarizationModule
def save_timeout_key(timeout_file, key):
    if os.path.exists(timeout_file):
        with open(timeout_file, 'r') as f:
            timeout_keys = json.load(f)
    else:
        timeout_keys = {}
    timeout_keys[str(len(timeout_keys))] = key
    with open(timeout_file, 'w') as f:
        json.dump(timeout_keys, f, indent=4)

def load_timeout_keys(timeout_file):
    if not os.path.exists(timeout_file):
        return set()
    with open(timeout_file, 'r') as f:
        timeout_keys = json.load(f)
    return set(timeout_keys.values())
def run_single_file(file_path, value, state, solve_info, q):
    try:
        with open(file_path, 'r') as file:
            smtlib_str = file.read()

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        try:
            smtlib_str, var_dict, constant_list = normalize_smt_str(smtlib_str)
        except MyException:
            logger.info(f'归一化超时')
            return
        finally:
            signal.alarm(0)

        if var_dict is None:
            return

        assertions = parse_smt2_string(smtlib_str)
        solver = Solver()
        for a in assertions:
            solver.add(a)

        result_list = [solve_info[0], solve_info[1], solve_info[2]]
        start_time = time.time()

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60 * 20)
        try:
            embedder = CodeEmbedder_normalize()
            set_seed(0)

            model = EnhancedClassifier()
            model.load_state_dict(torch.load('/home/lz/PycharmProjects/Pearl/test_rl/predictor/smt_comp_NIA/QF_NIA_bert_predictor_mask_best_llm.pth'))
            model.eval()

            model_time = EnhancedEightClassModelLargeInput()
            model_time.load_state_dict(torch.load('/home/lz/PycharmProjects/Pearl/test_rl/predictor/smt_comp_NIA/QF_NIA_bert_predictor_2_mask_best_model_llm.pth'))
            model_time.eval()

            env = ConstraintSimplificationEnv_test(embedder, assertions, model, model_time, smtlib_str,
                                                   file_path, var_dict, state)
            observation, action_space = env.reset()

            agent = PearlAgent(
                policy_learner=SoftActorCritic(
                    state_dim=8192,
                    action_space=action_space,
                    actor_hidden_dims=[1024, 512, 128],
                    critic_hidden_dims=[1024, 512, 128],
                    action_representation_module=IdentityActionRepresentationModule(
                        max_number_actions=action_space.n,
                        representation_dim=action_space.action_dim
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

            info = online_learning(
                agent=agent,
                env=env,
                number_of_episodes=1,
                print_every_x_episodes=1,
                record_period=1
            )
        except MyException:
            logger.info(f'执行超时: {file_path}')
        finally:
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
            q.put((file_path, result_list))
            torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"处理 {file_path} 时发生错误: {e}")


def test_group():
    timeout_file = 'timeout_keys.json'
    timeout_keys = load_timeout_keys(timeout_file)

    with open('/home/lz/PycharmProjects/Pearl/test_rl/test_solve/NIA/NIA.json', 'r') as file:
        solve_dict = json.load(file)

    info_name = 'info_dict_gai_6_normal_0503_pre_llm_llama3.1:70b_1200s_QF_NIA.txt'
    if not os.path.exists(info_name):
        with open(info_name, 'w') as file:
            json.dump({}, file, indent=4)
        info_dict = {}
    else:
        info_dict = load_dictionary(info_name)

    with open('/home/lz/PycharmProjects/Pearl/test_rl/predictor/smt_comp_NIA/QF_NIA_test.json', 'r') as file:
        result_dict = json.load(file)

    items = list(result_dict.items())
    random.shuffle(items)

    for key, value in items:
        state = torch.tensor(np.load(value[0]))
        solve_info = solve_dict.get(key)

        if not solve_info or solve_info[0] not in ["sat", "unknown"]:
            continue
        if solve_info[1] <= 300 or key in info_dict.keys():
            continue
        # if '20220315-MathProblems' in key:
        #     continue
        if key in timeout_keys:
            # logger.warning(f"跳过超时的文件: {key}")
            continue
        q = Queue()
        p = Process(target=run_single_file, args=(key, value, state, solve_info, q))
        p.start()
        p.join(timeout=60 * 25)
        if p.is_alive():
            logger.warning(f"子进程超时: {key}, 强制终止")
            save_timeout_key(timeout_file, key)
            p.terminate()
            p.join()
        else:
            if not q.empty():
                file_path, result_list = q.get()
                info_dict[file_path] = result_list
                with open(info_name, 'w') as file:
                    json.dump(info_dict, file, indent=4)


if __name__ == '__main__':
    test_group()
