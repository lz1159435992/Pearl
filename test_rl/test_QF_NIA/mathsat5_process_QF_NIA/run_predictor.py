import os
import sys
import json
import time

if '/home/lz/PycharmProjects/Pearl' not in sys.path:
    sys.path.insert(0, '/home/lz/PycharmProjects/Pearl')

from loguru import logger
from test_rl.test_QF_NIA.cvc5_process_QF_NIA import run_predictor as base_rp


def load_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'config.json not found at: {config_path}')
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _process_worker_mathsat(file_path, list1, embedding_path, shared_result, env_data_queue,
                             solver_name, llm_host, llm_model, models_dir, script_dir, timeout, time_bins):
    try:
        import torch
        import numpy as np
        import copy
        import time as _time
        from z3.z3 import parse_smt2_string, Solver as Z3_Solver
        from pearl.utils.functional_utils.experimentation.set_seed import set_seed
        from test_rl.test_QF_NIA.mathsat5_process_QF_NIA.train_predictor import EnhancedClassifier, EnhancedEightClassModelLargeInput
        from test_rl.test_script.utils import normalize_smt_str
        from test_rl.test_QF_NIA.mathsat5_process_QF_NIA.test_group_get_dis_smt_comp_bert_embeding_single import process_embeding

        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        # Keep base module's device in sync to avoid tensor device mismatch
        try:
            base_rp.device = device
        except Exception:
            pass

        result_list = [list1[0], list1[1], list1[2]]

        with open(file_path, 'r') as file:
            smtlib_str = file.read()
        smtlib_str, var_dict, constant_list = normalize_smt_str(smtlib_str)
        assertions = parse_smt2_string(smtlib_str)

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

        state = torch.tensor(np.load(embedding_path))

        embedder = None
        set_seed(0)
        z3_assertions = copy.deepcopy(assertions)
        env = base_rp.ConstraintSimplificationEnv_test(
            embedder, z3_assertions, model, model_time, smtlib_str,
            file_path, var_dict, state,
            solver_name=solver_name, llm_host=llm_host, llm_model=llm_model, global_timeout=timeout,
            time_dict=time_bins
        )

        env_data_queue.put({
            'total_solve_time': env.total_solve_time,
            'final_solve_time': env.solve_time,
            'llm_time': env.llm_time,
            'counterexamples_list': env.counterexamples_list,
        })

        observation, action_space = env.reset()
        agent = base_rp.create_agent(env, action_space)

        last_check_time = _time.time()
        check_interval = 5

        def data_callback(e, episode, step):
            nonlocal last_check_time
            now = _time.time()
            if now - last_check_time >= check_interval:
                env_data_queue.put({
                    'total_solve_time': e.total_solve_time,
                    'final_solve_time': e.solve_time,
                    'llm_time': e.llm_time,
                    'counterexamples_list': e.counterexamples_list,
                })
                last_check_time = now
            return False

        start_time = _time.time()
        info = base_rp.online_learning(
            agent=agent,
            env=env,
            number_of_episodes=1,
            print_every_x_episodes=1,
            record_period=1,
            callback=data_callback,
        )

        total_solve_time = env.total_solve_time
        final_solve_time = env.solve_time
        llm_total_time = env.llm_time
        counterexamples_list = env.counterexamples_list

        end_time = _time.time()
        total_execution_time = end_time - start_time

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
        shared_result['result'] = [list1[0], list1[1], list1[2], 0, 0, 0, 0, 'failed', [], [[]]]


def process_single_file_with_timeout_mathsat(file_path, list1, embedding_path, info_dict, info_name,
                                             solver_name, llm_host, llm_model, timeout, models_dir, script_dir, time_bins):
    from multiprocessing import get_context
    ctx = get_context('spawn')
    manager = ctx.Manager()
    shared_result = manager.dict()
    env_data_queue = ctx.Queue()

    p = ctx.Process(target=_process_worker_mathsat, args=(
        file_path, list1, embedding_path, shared_result, env_data_queue,
        solver_name, llm_host, llm_model, models_dir, script_dir, timeout, time_bins
    ))
    p.start()

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
        info_dict[file_path] = result_list
        try:
            tmp_path = info_name + '.tmp'
            with open(tmp_path, 'w') as file:
                json.dump(info_dict, file, indent=4)
            os.replace(tmp_path, info_name)
        except Exception as e:
            logger.error(f'写入结果文件失败: {e}')
        return result_list, 'failed'

    result = shared_result.get('result', None)
    if result is None:
        result = [list1[0], list1[1], list1[2], 0, 0, 0, 0, 'failed', [], [[]]]
        status = 'failed'
    else:
        status = 'succeed' if result[7] == 'succeed' else 'failed'
    info_dict[file_path] = result
    try:
        tmp_path = info_name + '.tmp'
        with open(tmp_path, 'w') as file:
            json.dump(info_dict, file, indent=4)
        os.replace(tmp_path, info_name)
    except Exception as e:
        logger.error(f'写入结果文件失败: {e}')
    return result, status


def main():
    config = load_config()
    try:
        base_rp.__file__ = __file__
    except Exception:
        pass

    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, 'log')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'run_predictor.log')
    try:
        logger.remove()
    except Exception:
        pass
    logger.add(log_file_path, enqueue=True, rotation='100 MB', retention='14 days', encoding='utf-8')

    def patched_psf(file_path, list1, embedding_path, info_dict, info_name,
                    solver_name, llm_host, llm_model, timeout, models_dir, script_dir, time_bins):
        return process_single_file_with_timeout_mathsat(
            file_path, list1, embedding_path, info_dict, info_name,
            solver_name, llm_host, llm_model, timeout, models_dir, script_dir, time_bins
        )
    base_rp.process_single_file_with_timeout = patched_psf

    default_llm_host = config.get('embedding', {}).get('host')
    default_llm_model = config.get('embedding', {}).get('model')
    default_timeout = config.get('solver', {}).get('timeout', 1200)

    import argparse
    parser = argparse.ArgumentParser(description='运行QF_NIA MathSAT5求解实验')
    parser.add_argument('--llm_host', type=str, default=default_llm_host)
    parser.add_argument('--llm_model', type=str, default=default_llm_model)
    parser.add_argument('--timeout', type=int, default=default_timeout)
    parser.add_argument('--max_files', type=int, default=None)
    args = parser.parse_args()

    base_rp.run_QF_NIA_experiment(
        solver_name='mathsat',
        llm_host=args.llm_host,
        llm_model=args.llm_model,
        timeout=args.timeout,
        max_files=args.max_files,
        config=config,
    )


if __name__ == '__main__':
    main()
