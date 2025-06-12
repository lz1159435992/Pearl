# Welcome to Cursor
from test_rl.test_script.search_test import load_dictionary
from test_rl.test_script.utils import setup_logger
from loguru import logger
# Welcome to Cursor

# 1. Try generating with command K on a new line. Ask for a pytorch script of a feedforward neural network
# 2. Then, select the outputted code and hit chat. Ask if there's a bug. Ask how to improve.
# 3. Try selecting some code and hitting edit. Ask the bot to add residual layers.
# 4. To try out cursor on your own projects, go to the file menu (top left) and open a folder.

print("Hello, World!")

import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json

class SolverResult:
    def __init__(self, solve_time, result, model):
        self.solve_time = solve_time
        self.result = result
        self.model = model

class Solver:
    def solve(self, smtlib_str, timeout=5):
        raise NotImplementedError

class CVC5Solver(Solver):
    def solve(self, smtlib_str, timeout=5):
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.smt2', delete=False) as f:
            f.write(smtlib_str)
            f.flush()
            cmd = ['cvc5', '--lang', 'smt2', '--produce-models', f.name, f'--tlimit={int(timeout*1000)}']
            start = time.time()
            logger.info(f"Running command: {' '.join(cmd)}")
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
        return SolverResult(elapsed, result, model)

class Z3Solver(Solver):
    def solve(self, smtlib_str, timeout=5):
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.smt2', delete=False) as f:
            f.write(smtlib_str)
            f.flush()
            cmd = ['z3', f'-T:{timeout}', f.name]
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
        return SolverResult(elapsed, result, model)

class MathSAT5Solver(Solver):
    def solve(self, smtlib_str, timeout=5):
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.smt2', delete=False) as f:
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
        return SolverResult(elapsed, result, model)

class OpenSMTSolver(Solver):
    def solve(self, smtlib_str, timeout=5):
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.smt2', delete=False) as f:
            f.write(smtlib_str)
            f.flush()
            cmd = ['opensmt', f.name]
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
        return SolverResult(elapsed, result, model)

class YicesSolver(Solver):
    def solve(self, smtlib_str, timeout=5):
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.smt2', delete=False) as f:
            f.write(smtlib_str)
            f.flush()
            cmd = ['yices-smt2', f.name, f'--timeout={timeout}']
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
        return SolverResult(elapsed, result, model)

def concurrent_solve(solver, smtlib_list, timeout=5, max_workers=4):
    results = [None] * len(smtlib_list)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {

            executor.submit(solver.solve, smtlib, timeout): idx
            for idx, smtlib in enumerate(smtlib_list)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = SolverResult(0, 'error', str(e))
    return results

def parse_model_to_dict(model_str):
    """
    尝试将求解器输出的模型字符串转为字典，简单处理（可根据实际输出格式扩展）
    """
    model = {}
    if not model_str:
        return model
    lines = model_str.splitlines()
    for line in lines:
        if line.strip().startswith('(define-fun'):
            parts = line.strip().split()
            if len(parts) >= 4:
                var = parts[1]
                # 尝试获取下一个非括号的值
                for v in parts[3:]:
                    if v not in ('(', ')'):
                        model[var] = v
                        break
    return model

def load_result_dict(json_path):
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_result_dict(json_path, result_dict):
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)

def solve_files_with_all_solvers(file_paths, timeout=5, save_dir="solver_results"):
    solvers = [
        ("CVC5", CVC5Solver()),
        ("Z3", Z3Solver()),
        ("MathSAT5", MathSAT5Solver()),
        ("OpenSMT", OpenSMTSolver()),
        ("Yices", YicesSolver()),
    ]
    for file_path in file_paths:
        print(f"\n=== Solving file: {file_path} ===")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                smtlib_str = f.read()
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
            continue
        for name, solver in solvers:
            print(f"\n--- Using {name} ---")
            try:
                result = solver.solve(smtlib_str, timeout=timeout)
                print("  Time:", result.solve_time)
                print("  Result:", result.result)
                print("  Model:", result.model)
                save_solver_result(name, file_path, result.result, result.solve_time, timeout, result.model, save_dir)
            except Exception as e:
                print(f"  Error running {name}: {e}")

# 示例用法
if __name__ == "__main__":
    setup_logger()
    file_paths = []
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
    import random
    random.shuffle(items)  # 打乱顺序，增加随机性
    for key, value in items:
        solve_info = solve_dict.get(key)
        if not solve_info or solve_info[0] not in ["sat", "unknown"]:
            continue
        if solve_info[1] <= 300 or key in info_dict.keys():
            continue
        file_paths.append(key)
    solver_name = "CVC5"
    solver = CVC5Solver()
    timeout =1200  # 设置超时时间为1200秒
    json_path = f"/home/lz/PycharmProjects/Pearl/test_rl/test_cvc5/QF_NIA_results/CVC5_results.json"

    # 1. 读取已有结果
    result_dict = load_result_dict(json_path)

    # 2. 筛选未求解的文件
    unsolved_paths = []
    unsolved_smtlib = []
    for path in file_paths:
        if path in result_dict:
            print(f"{solver_name}: {path} 已存在，跳过。")
        else:
            with open(path, 'r', encoding='utf-8') as f:
                unsolved_paths.append(path)
                unsolved_smtlib.append(f.read())

    # 3. 并发求解未求解的文件
    if unsolved_smtlib:
        results = concurrent_solve(solver, unsolved_smtlib, timeout=timeout, max_workers=3)
        for i, result in enumerate(results):
            model_dict = parse_model_to_dict(result.model)
            result_dict[unsolved_paths[i]] = [
                result.result,
                result.solve_time,
                timeout,
                model_dict
            ]
        # 4. 保存累加后的结果
        save_result_dict(json_path, result_dict)
        print(f"新结果已保存到 {json_path}")
    else:
        print("所有文件均已存在结果，无需重复求解。")

    # 5. 可选：输出所有结果
    for path, value in result_dict.items():
        print(f"{path}: {value}")

    # info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1110_pre_SMTimer_llama3.1:70b_1200s_info_dict_rl.txt'
    # # info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1109_pre_SMTimer_llama3.1:70b_1200s.txt'
    #
    # info_dict = load_dictionary(info_name)
    # file_paths = info_dict.keys()
    # if file_paths:
    #     solve_files_with_all_solvers(file_paths, timeout=1200, save_dir="smtimer_cvc5_results")
    # else:
    #     print("请在file_paths中填写你的smt2文件路径进行测试。")

    # file_paths = []
    # with open('/home/lz/PycharmProjects/Pearl/test_rl/test_solve/NIA/NIA.json', 'r') as file:
    #     solve_dict = json.load(file)
    #
    # info_name = 'info_dict_gai_6_normal_0503_pre_llm_llama3.1:70b_1200s_QF_NIA.txt'
    # if not os.path.exists(info_name):
    #     with open(info_name, 'w') as file:
    #         json.dump({}, file, indent=4)
    #     info_dict = {}
    # else:
    #     info_dict = load_dictionary(info_name)
    #
    # with open('/home/lz/PycharmProjects/Pearl/test_rl/predictor/smt_comp_NIA/QF_NIA_test.json', 'r') as file:
    #     result_dict = json.load(file)
    #
    # items = list(result_dict.items())
    #
    #
    # for key, value in items:
    #     solve_info = solve_dict.get(key)
    #     if not solve_info or solve_info[0] not in ["sat", "unknown"]:
    #         continue
    #     if solve_info[1] <= 300 or key in info_dict.keys():
    #         continue
    #     file_paths.append(key)
    # if file_paths:
    #     solve_files_with_all_solvers(file_paths, timeout=1200, save_dir="QF_NIA_results")
    # else:
    #     print("请在file_paths中填写你的smt2文件路径进行测试。")
