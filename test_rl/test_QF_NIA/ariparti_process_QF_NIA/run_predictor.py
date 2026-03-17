import os
import sys
import json
import time
import tempfile
import subprocess
import math
from pathlib import Path

# Ensure repo root on sys.path so that absolute package import works
if '/home/lz/PycharmProjects/Pearl' not in sys.path:
    sys.path.insert(0, '/home/lz/PycharmProjects/Pearl')

# Ensure AriParti paths and binaries available
try:
    _cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    if os.path.exists(_cfg_path):
        with open(_cfg_path, 'r', encoding='utf-8') as _f:
            _cfg = json.load(_f)
        _home = (_cfg.get('ariparti', {}) or {}).get('home')
        if _home:
            os.environ.setdefault('ARIPARTI_HOME', _home)
except Exception:
    pass

import project_bootstrap  # sets ARIPARTI sys.path/PATH
from loguru import logger

# Reuse the full experiment pipeline from cvc5 version
from test_rl.test_QF_NIA.cvc5_process_QF_NIA import run_predictor as base_rp


def load_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'config.json not found at: {config_path}')
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


class AriPartiSolver:
    def __init__(self, cfg: dict):
        self.cfg = cfg or {}
        ariparti_cfg = self.cfg.get('ariparti', {})
        home = ariparti_cfg.get('home') or os.environ.get('ARIPARTI_HOME') or '/home/lz/PycharmProjects/AriParti'
        self.ariparti_home = home
        part_path = ariparti_cfg.get('partitioner_path', 'bin/partitioner')
        solver_path = ariparti_cfg.get('solver_path', 'bin/linux-prebuilt/base-solvers/cvc5-1.0.8')
        self.max_running_tasks = int(ariparti_cfg.get('max_running_tasks', 8))
        # absolutize
        self.partitioner_path = part_path if os.path.isabs(part_path) else os.path.join(home, part_path)
        self.base_solver_path = solver_path if os.path.isabs(solver_path) else os.path.join(home, solver_path)
        self.ariparti_entry = os.path.join(home, 'src', 'AriParti.py')
        if not os.path.exists(self.ariparti_entry):
            raise FileNotFoundError(f'AriParti.py not found at {self.ariparti_entry}. Set ARIPARTI_HOME or config.ariparti.home correctly.')

    def solve(self, smtlib_str: str, timeout: float = 5):
        start = time.time()
        # Prepare temp files/dirs
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_smt = os.path.join(tmpdir, 'input.smt2')
            with open(tmp_smt, 'w', encoding='utf-8') as f:
                f.write(smtlib_str)
            out_dir = os.path.join(tmpdir, 'out')
            os.makedirs(out_dir, exist_ok=True)

            if timeout and timeout > 0:
                time_limit_sec = int(math.ceil(float(timeout)))
                if time_limit_sec <= 0:
                    time_limit_sec = 1
                proc_timeout = time_limit_sec + 60
            else:
                time_limit_sec = 0
                proc_timeout = None

            cmd = [
                sys.executable, self.ariparti_entry,
                '--file', tmp_smt,
                '--output-dir', out_dir,
                '--partitioner', self.partitioner_path,
                '--solver', self.base_solver_path,
                '--max-running-tasks', str(self.max_running_tasks),
                '--time-limit', str(time_limit_sec),
            ]
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=proc_timeout,
                )
                elapsed = time.time() - start
                stdout = (proc.stdout or '').strip()
                stderr = (proc.stderr or '').strip()
                if proc.returncode == 0:
                    lines = stdout.splitlines()
                    if len(lines) >= 2:
                        result = lines[0].strip()
                        try:
                            solve_time = float(lines[1].strip())
                        except Exception:
                            solve_time = elapsed
                        # Map to our expected labels
                        if result not in ['sat', 'unsat', 'unknown', 'timeout']:
                            result = 'unknown'
                        return base_rp.SolverResult(solve_time, result, None)
                    else:
                        logger.error(f'AriParti output format error: {stdout}\nStderr: {stderr}')
                        return base_rp.SolverResult(elapsed, 'error', None)
                else:
                    logger.error(f'AriParti returned non-zero {proc.returncode}. Stdout: {stdout}\nStderr: {stderr}')
                    return base_rp.SolverResult(elapsed, 'error', None)
            except subprocess.TimeoutExpired:
                elapsed = time.time() - start
                return base_rp.SolverResult(elapsed, 'timeout', None)
            except Exception as e:
                elapsed = time.time() - start
                logger.exception(f'AriParti invocation failed: {e}')
                return base_rp.SolverResult(elapsed, 'error', None)


def _process_worker_ariparti(file_path, list1, embedding_path, shared_result, env_data_queue,
                             solver_name, llm_host, llm_model, models_dir, script_dir, timeout, time_bins, config):
    try:
        # Device in child
        import torch
        import numpy as np
        import copy
        import time as _time
        import traceback as _traceback
        from z3.z3 import parse_smt2_string, Solver as Z3_Solver
        from pearl.utils.functional_utils.experimentation.set_seed import set_seed
        from test_rl.test_QF_NIA.ariparti_process_QF_NIA.train_predictor import EnhancedClassifier, EnhancedEightClassModelLargeInput
        from test_rl.test_script.utils import normalize_smt_str
        from test_rl.test_QF_NIA.ariparti_process_QF_NIA.test_group_get_dis_smt_comp_bert_embeding_single import process_embeding

        # patch base get_solver in child (preserve original to avoid recursion)
        ariparti_solver_singleton = AriPartiSolver(config)
        _orig_get_solver_child = base_rp.get_solver

        def patched_get_solver(name: str):
            if name == 'ariparti':
                return ariparti_solver_singleton
            return _orig_get_solver_child(name)

        base_rp.get_solver = patched_get_solver

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        result_list = [list1[0], list1[1], list1[2]]

        # read + normalize
        with open(file_path, 'r') as file:
            smtlib_str = file.read()
        smtlib_str, var_dict, constant_list = normalize_smt_str(smtlib_str)
        assertions = parse_smt2_string(smtlib_str)

        # load models
        model = EnhancedClassifier(input_dim=8192)
        model_path = os.path.join(script_dir, models_dir, 'QF_NIA_bert_predictor_mask_best.pth')
        import torch as _torch
        model.load_state_dict(_torch.load(model_path, map_location=_torch.device('cpu')))
        model.to(device)
        model.eval()

        model_time = EnhancedEightClassModelLargeInput(input_dim=8192)
        model_time_path = os.path.join(script_dir, models_dir, 'QF_NIA_bert_predictor_2_mask_best_model.pth')
        model_time.load_state_dict(_torch.load(model_time_path, map_location=_torch.device('cpu')))
        model_time.to(device)
        model_time.eval()

        # initial state
        state = _torch.tensor(np.load(embedding_path))

        embedder = None
        set_seed(0)
        z3_assertions = copy.deepcopy(assertions)
        env = base_rp.ConstraintSimplificationEnv_test(
            embedder, z3_assertions, model, model_time, smtlib_str,
            file_path, var_dict, state,
            solver_name=solver_name, llm_host=llm_host, llm_model=llm_model, global_timeout=timeout,
            time_dict=time_bins
        )

        # send initial data
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
        try:
            log_dir = os.path.join(script_dir, 'log')
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, 'worker_errors.log')
            with open(log_path, 'a', encoding='utf-8') as logf:
                logf.write(f"[{_time.strftime('%Y-%m-%d %H:%M:%S')}] file={file_path} solver={solver_name} host={llm_host} model={llm_model}\n")
                logf.write(f"embedding_path={embedding_path} models_dir={models_dir}\n")
                logf.write(f"Exception: {repr(e)}\n")
                import traceback as __tb
                logf.write(__tb.format_exc())
                logf.write('\n' + ('-'*80) + '\n')
        except Exception:
            pass
        shared_result['result'] = [list1[0], list1[1], list1[2], 0, 0, 0, 0, 'failed', [], [[]]]


def process_single_file_with_timeout_ariparti(file_path, list1, embedding_path, info_dict, info_name,
                                              solver_name, llm_host, llm_model, timeout, models_dir, script_dir, time_bins, config):
    from multiprocessing import get_context
    ctx = get_context('spawn')
    manager = ctx.Manager()
    shared_result = manager.dict()
    env_data_queue = ctx.Queue()

    p = ctx.Process(target=_process_worker_ariparti, args=(
        file_path, list1, embedding_path, shared_result, env_data_queue,
        solver_name, llm_host, llm_model, models_dir, script_dir, timeout, time_bins, config
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
    # Load our config and patch base pipeline
    config = load_config()

    try:
        base_rp.__file__ = __file__
    except Exception:
        pass

    # Reconfigure logger to write into this module's log dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, 'log')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'run_predictor.log')
    logger.remove()
    logger.add(log_file_path, enqueue=True, rotation='100 MB', retention='14 days', encoding='utf-8')

    # data.source accepts either:
    # - Flat mapping: {file_path: [status, solve_time, timeout, extra]}
    # - AriParti_sync nested format: {"metadata":..., "results": {file_path: {"result":..., "solve_time":...}}}
    # Convert only when nested format is detected.
    try:
        src_path = config.get('data', {}).get('source')
        if src_path and os.path.exists(src_path):
            with open(src_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, dict) and isinstance(data.get('results', None), dict):
                results = data.get('results', {})
                solve_map = {}
                for fpath, entry in results.items():
                    if not isinstance(entry, dict):
                        continue
                    status = (entry.get('result') or 'unknown').lower()
                    if status in ['timeout', 'error']:
                        status = 'unknown'
                    t = entry.get('solve_time', 0) or 0
                    try:
                        t = float(t)
                    except Exception:
                        t = 0.0
                    timeout_val = config.get('solver', {}).get('timeout', 1200)
                    solve_map[fpath] = [status, t, timeout_val, None]

                script_dir = os.path.dirname(os.path.abspath(__file__))
                tmp_map_path = os.path.join(script_dir, 'ariparti_solve_map.json')
                with open(tmp_map_path, 'w', encoding='utf-8') as wf:
                    json.dump(solve_map, wf, ensure_ascii=False, indent=2)
                config['data']['source'] = tmp_map_path
    except Exception as e:
        logger.warning(f'解析/转换AriParti结果失败，将继续使用原data.source: {e}')

    # Monkey-patch base process function to use AriParti worker (so child uses our solver)
    def patched_psf(file_path, list1, embedding_path, info_dict, info_name,
                    solver_name, llm_host, llm_model, timeout, models_dir, script_dir, time_bins):
        return process_single_file_with_timeout_ariparti(
            file_path, list1, embedding_path, info_dict, info_name,
            solver_name, llm_host, llm_model, timeout, models_dir, script_dir, time_bins, config
        )
    base_rp.process_single_file_with_timeout = patched_psf

    # Also patch get_solver in main process to support 'ariparti' (preserve original)
    try:
        _ariparti_solver_singleton_main = AriPartiSolver(config)
        _orig_get_solver_main = base_rp.get_solver

        def _patched_get_solver_main(name: str):
            if name == 'ariparti':
                return _ariparti_solver_singleton_main
            return _orig_get_solver_main(name)

        base_rp.get_solver = _patched_get_solver_main
    except Exception as e:
        logger.warning(f'主进程替换get_solver失败: {e}')

    # Parse CLI (reusing defaults from our config)
    default_llm_host = config.get('embedding', {}).get('host')
    default_llm_model = config.get('embedding', {}).get('model')
    default_timeout = config.get('solver', {}).get('timeout', 1200)

    import argparse
    parser = argparse.ArgumentParser(description='运行QF_NIA AriParti求解实验')
    parser.add_argument('--llm_host', type=str, default=default_llm_host)
    parser.add_argument('--llm_model', type=str, default=default_llm_model)
    parser.add_argument('--timeout', type=int, default=default_timeout)
    parser.add_argument('--max_files', type=int, default=None)
    args = parser.parse_args()

    # Run the base experiment pipeline with our config and solver
    base_rp.run_QF_NIA_experiment(
        solver_name='ariparti',
        llm_host=args.llm_host,
        llm_model=args.llm_model,
        timeout=args.timeout,
        max_files=args.max_files,
        config=config,
    )


if __name__ == '__main__':
    main()
