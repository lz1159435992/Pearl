# Welcome to Cursor
from test_rl.test_script.search_test import load_dictionary
from test_rl.test_script.time import file_path

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

def save_solver_result(solver_name, file_path, result, elapsed, timeout, model, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, f'{solver_name}_results.json')
    # 读取已存在的结果
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            results_dict = json.load(f)
    else:
        results_dict = {}
    # 跳过已存在的
    if file_path in results_dict:
        print(f"  {solver_name}: {file_path} 已存在，跳过。")
        return
    # 处理模型
    model_dict = parse_model_to_dict(model)
    # 保存新结果
    results_dict[file_path] = [result, elapsed, timeout, model_dict]
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)
    print(f"  {solver_name}: 结果已保存到 {json_path}")

def solve_files_with_all_solvers(file_paths, timeout=5, save_dir="solver_results"):
    solvers = [
        ("CVC5", CVC5Solver()),
        # ("Z3", Z3Solver()),
        # ("MathSAT5", MathSAT5Solver()),
        # ("OpenSMT", OpenSMTSolver()),
        # ("Yices", YicesSolver()),
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
    # info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1110_pre_SMTimer_llama3.1:70b_1200s_info_dict_rl.txt'
    # # info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1109_pre_SMTimer_llama3.1:70b_1200s.txt'
    #
    # info_dict = load_dictionary(info_name)
    # file_paths = info_dict.keys()
    # if file_paths:
    #     solve_files_with_all_solvers(file_paths, timeout=1200, save_dir="smtimer_cvc5_results")
    # else:
    #     print("请在file_paths中填写你的smt2文件路径进行测试。")

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
    random.shuffle(items)

    for key, value in items:
        solve_info = solve_dict.get(key)
        if not solve_info or solve_info[0] not in ["sat", "unknown"]:
            continue
        if solve_info[1] <= 300 or key in info_dict.keys():
            continue
        file_paths.append(key)
    if file_paths:
        solve_files_with_all_solvers(file_paths, timeout=1200, save_dir="QF_NIA_results")
    else:
        print("请在file_paths中填写你的smt2文件路径进行测试。")
