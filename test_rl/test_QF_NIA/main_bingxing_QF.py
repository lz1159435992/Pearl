# Welcome to Cursor
from test_rl.test_script.search_test import load_dictionary
from test_rl.test_script.utils import setup_logger, solve_and_measure_time, model_to_dict
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
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import os
import json
from z3.z3 import parse_smt2_string, Solver as Z3_Solver, sat, unsat, unknown
import multiprocessing
import re

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
            logger.info(f"Running command: {' '.join(cmd)}")
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

class MathSAT5Solver(Solver):
    def solve(self, smtlib_str, timeout=5):
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.smt2', delete=False) as f:
            f.write(smtlib_str)
            f.flush()
            cmd = ['mathsat', f.name]
            logger.info(f"Running command: {' '.join(cmd)}")
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
            logger.info(f"Running command: {' '.join(cmd)}")
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
            cmd = ['yices-smt2', f.name, f'--timeout={int(timeout*1000)}']
            logger.info(f"Running command: {' '.join(cmd)}")
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

def solve_with_solver_name(solver_name, smtlib_str, timeout):
    if solver_name.lower() == "z3":
        solver = Z3Solver()
    elif solver_name.lower() == "yices":
        solver = YicesSolver()
    else:
        raise ValueError(f"Unknown solver: {solver_name}")
    result = solver.solve(smtlib_str, timeout)
    return {
        "result": result.result,
        "solve_time": result.solve_time,
        "timeout": timeout,
        "model": result.model
    }

def concurrent_solve_process(solver_name, smtlib_list, timeout=5, max_workers=4):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    results = [None] * len(smtlib_list)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(solve_with_solver_name, solver_name, smtlib, timeout): idx
            for idx, smtlib in enumerate(smtlib_list)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = {
                    "result": "error",
                    "solve_time": 0,
                    "timeout": timeout,
                    "model": str(e)
                }
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

def process_z3_file(args):
    file_path, timeout = args
    import json
    from z3.z3 import parse_smt2_string, Solver
    from test_rl.test_script.utils import solve_and_measure_time, model_to_dict
    from loguru import logger
    try:
        logger.info(f"[Worker] 开始处理: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            smtlib_str = f.read()
        try:
            dict_obj = json.loads(smtlib_str)
            if 'smt-comp' in file_path:
                smtlib_str = dict_obj['smt_script']
            else:
                smtlib_str = dict_obj['script']
        except json.JSONDecodeError:
            pass  # 不是json则直接用原始内容
        try:
            assertions = parse_smt2_string(smtlib_str)
        except Exception as e:
            logger.info(f"[Worker] 解析SMT-LIB失败: {file_path}, 错误: {e}")
            return (file_path, ['parse_error', 0, timeout, str(e)])
        solver = Solver()
        if isinstance(assertions, list):
            solver.add(*assertions)
        else:
            solver.add(assertions)
        result, model, time_taken = solve_and_measure_time(solver, timeout)
        result_list = [result, time_taken, timeout]
        if model:
            result_list.append(model_to_dict(model))
        else:
            result_list.append('No model')
        logger.info(f"[Worker] 完成: {file_path}, 结果: {result}, 用时: {time_taken:.2f}s")
        return (file_path, result_list)
    except Exception as e:
        logger.info(f"[Worker] 处理异常: {file_path}, 错误: {e}")
        return (file_path, ['error', 0, timeout, str(e)])

def get_Z3_result():
    setup_logger()
    info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1110_pre_SMTimer_llama3.1:70b_1200s_info_dict_rl.txt'
    info_dict = load_dictionary(info_name)
    file_paths = list(info_dict.keys())
    solver_name = "z3"
    timeout = 1200  # 设置超时时间为1200秒
    json_path = f"{solver_name.lower()}_smtimer_results.json"

    logger.info(f"Z3批量求解任务启动，待处理文件数: {len(file_paths)}")
    # 1. 读取已有结果
    result_dict = load_result_dict(json_path)

    # 2. 筛选未求解的文件
    unsolved_paths = [p for p in file_paths if p not in result_dict]
    logger.info(f"未求解文件数: {len(unsolved_paths)}")
    if not unsolved_paths:
        logger.info("所有文件均已存在结果，无需重复求解。")
        print("所有文件均已存在结果，无需重复求解。")
        return

    # 3. 多进程并发求解
    tasks = [(path, timeout*1000) for path in unsolved_paths]
    with multiprocessing.Pool(processes=4) as pool:
        for file_path, result_list in pool.imap_unordered(process_z3_file, tasks):
            result_dict[file_path] = result_list
            logger.info(f"主进程收集结果: {file_path}, 状态: {result_list[0]}")
            print(f"已完成: {file_path}, 结果: {result_list[0]}")

    # 4. 保存累加后的结果
    save_result_dict(json_path, result_dict)
    logger.info(f"全部结果已保存到 {json_path}")
    print(f"新结果已保存到 {json_path}")

def get_CVC5_result():
    file_paths = []
    info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1110_pre_SMTimer_llama3.1:70b_1200s_info_dict_rl.txt'
    info_dict = load_dictionary(info_name)
    file_paths = info_dict.keys()
    solver_name = "CVC5"
    solver = CVC5Solver()
    timeout = 1200  # 设置超时时间为1200秒
    json_path = f"{solver_name.lower()}_smtimer_results.json"

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

                smtlib_str = f.read()
                # 解析字符串
                try:
                    # 将JSON字符串转换为字典
                    dict_obj = json.loads(smtlib_str)
                    # print("转换后的字典：", dict_obj)
                except json.JSONDecodeError as e:
                    print("解析错误：", e)
                #
                if 'smt-comp' in path:
                    smtlib_str = dict_obj['smt_script']
                else:
                    smtlib_str = dict_obj['script']
                unsolved_smtlib.append(smtlib_str)

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
def get_CVC5_result_QF_NIA():
        # 2. 筛选未求解的文件
    unsolved_paths = []
    unsolved_smtlib = []
    solver_name = "CVC5"
    solver = CVC5Solver()
    timeout = 1200  # 设置超时时间为1200秒
    # 将JSON文件保存在脚本所在的目录下
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, f"{solver_name.lower()}_QF_NIA.json")
    
    # 获取CPU逻辑核心数
    cpu_cores = multiprocessing.cpu_count()
    max_workers = cpu_cores - 1
    
    # 1. 读取已有结果
    result_dict = load_result_dict(json_path)
    
    test_path = []
    directory = '/home/lz/Downloads/non-incremental_Hierarchy/non-incremental'
    test_path.append(directory)
    
    search_list = ['QF_NIA']
    
    # 遍历目录
    pattern = re.compile(r'\(set-info :status (\w+)\)')
    count = 0
    
    for directory in test_path:
        for search in search_list:
            for dirpath, dirnames, filenames in os.walk(os.path.join(directory, search)):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    
                    # 跳过特定文件
                    if any(skip in file_path for skip in ['starexec_description.txt', '52759_b3ecd2335fd16ec2eee2_9_UFDTBV', 'sll-optional-1.i_1']):
                        continue
                    
                    # 检查是否已存在结果
                    if file_path in result_dict:
                        continue
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            smtlib_str = file.read()
                        
                        match = pattern.search(smtlib_str)
                        if match:
                            status = match.group(1)
                            print(f"文件 {file_path} 状态: {status}")
                        else:
                            print(f"文件 {file_path} 未找到状态信息")
                            continue
                        
                        if status != 'unsat':
                            count += 1
                            unsolved_paths.append(file_path)  # ✅ 修复：使用file_path
                            unsolved_smtlib.append(smtlib_str)
                    
                    except Exception as e:
                        print(f"读取文件 {file_path} 时出错: {e}")
                        continue
    
    # 3. 并发求解未求解的文件
    if unsolved_smtlib:
        print(f"开始求解 {len(unsolved_smtlib)} 个文件...")
        results = concurrent_solve(solver, unsolved_smtlib, timeout=timeout, max_workers=max_workers)
        
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
    print(f"总共处理了 {len(result_dict)} 个文件")

def get_MathSAT_result_QF_NIA():
    """
    使用MathSAT5求解器批量处理QF_NIA问题
    支持断点续传功能，如果执行中断，下次执行会自动跳过已处理的文件
    """
    # 2. 筛选未求解的文件
    unsolved_paths = []
    unsolved_smtlib = []
    solver_name = "MathSAT5"
    solver = MathSAT5Solver()
    timeout = 1200  # 设置超时时间为1200秒
    # 将JSON文件保存在脚本所在的目录下
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, f"{solver_name.lower()}_QF_NIA.json")
    
    # 获取CPU逻辑核心数
    cpu_cores = multiprocessing.cpu_count()
    max_workers = cpu_cores - 1
    
    # 1. 读取已有结果
    result_dict = load_result_dict(json_path)
    
    test_path = []
    directory = '/home/lz/Downloads/non-incremental_Hierarchy/non-incremental'
    test_path.append(directory)
    
    search_list = ['QF_NIA']
    
    # 遍历目录
    pattern = re.compile(r'\(set-info :status (\w+)\)')
    count = 0
    
    for directory in test_path:
        for search in search_list:
            for dirpath, dirnames, filenames in os.walk(os.path.join(directory, search)):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    
                    # 跳过特定文件
                    if any(skip in file_path for skip in ['starexec_description.txt', '52759_b3ecd2335fd16ec2eee2_9_UFDTBV', 'sll-optional-1.i_1']):
                        continue
                    
                    # 检查是否已存在结果（断点续传机制）
                    if file_path in result_dict:
                        continue
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            smtlib_str = file.read()
                        
                        match = pattern.search(smtlib_str)
                        if match:
                            status = match.group(1)
                            print(f"文件 {file_path} 状态: {status}")
                        else:
                            print(f"文件 {file_path} 未找到状态信息")
                            continue
                        
                        # 只处理状态不为unsat的文件
                        if status != 'unsat':
                            count += 1
                            unsolved_paths.append(file_path)
                            unsolved_smtlib.append(smtlib_str)
                    
                    except Exception as e:
                        print(f"读取文件 {file_path} 时出错: {e}")
                        continue
    
    # 3. 并发求解未求解的文件
    if unsolved_smtlib:
        print(f"开始使用MathSAT5求解 {len(unsolved_smtlib)} 个文件...")
        results = concurrent_solve(solver, unsolved_smtlib, timeout=timeout, max_workers=max_workers)
        
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
        print(f"MathSAT5新结果已保存到 {json_path}")
    else:
        print("所有文件均已存在结果，无需重复求解。")
    
    # 5. 输出统计信息
    print(f"MathSAT5总共处理了 {len(result_dict)} 个文件")

def get_CVC5_result_QF_LIA():
    """
    使用CVC5求解器批量处理QF_LIA问题
    支持断点续传功能，如果执行中断，下次执行会自动跳过已处理的文件
    """
    # 2. 筛选未求解的文件
    unsolved_paths = []
    unsolved_smtlib = []
    solver_name = "CVC5"
    solver = CVC5Solver()
    timeout = 1200  # 设置超时时间为1200秒
    # 将JSON文件保存在脚本所在的目录下
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, f"{solver_name.lower()}_QF_LIA.json")
    
    # 获取CPU逻辑核心数
    cpu_cores = multiprocessing.cpu_count()
    max_workers = cpu_cores - 1
    
    # 1. 读取已有结果
    result_dict = load_result_dict(json_path)
    
    test_path = []
    directory = '/home/lz/Downloads/non-incremental_Hierarchy/non-incremental'
    test_path.append(directory)
    
    search_list = ['QF_LIA']
    
    # 遍历目录
    pattern = re.compile(r'\(set-info :status (\w+)\)')
    count = 0
    
    for directory in test_path:
        for search in search_list:
            for dirpath, dirnames, filenames in os.walk(os.path.join(directory, search)):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    
                    # 跳过特定文件
                    if any(skip in file_path for skip in ['starexec_description.txt', '52759_b3ecd2335fd16ec2eee2_9_UFDTBV', 'sll-optional-1.i_1']):
                        continue
                    
                    # 检查是否已存在结果（断点续传机制）
                    if file_path in result_dict:
                        continue
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            smtlib_str = file.read()
                        
                        match = pattern.search(smtlib_str)
                        if match:
                            status = match.group(1)
                            print(f"文件 {file_path} 状态: {status}")
                        else:
                            print(f"文件 {file_path} 未找到状态信息")
                            continue
                        
                        # 只处理状态不为unsat的文件
                        if status != 'unsat':
                            count += 1
                            unsolved_paths.append(file_path)
                            unsolved_smtlib.append(smtlib_str)
                    
                    except Exception as e:
                        print(f"读取文件 {file_path} 时出错: {e}")
                        continue
    
    # 3. 并发求解未求解的文件
    if unsolved_smtlib:
        print(f"开始使用CVC5求解 {len(unsolved_smtlib)} 个QF_LIA文件...")
        results = concurrent_solve(solver, unsolved_smtlib, timeout=timeout, max_workers=max_workers)
        
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
        print(f"CVC5新结果已保存到 {json_path}")
    else:
        print("所有文件均已存在结果，无需重复求解。")
    
    # 5. 输出统计信息
    print(f"CVC5总共处理了 {len(result_dict)} 个QF_LIA文件")

def get_MathSAT_result_QF_LIA():
    """
    使用MathSAT5求解器批量处理QF_LIA问题
    支持断点续传功能，如果执行中断，下次执行会自动跳过已处理的文件
    """
    # 2. 筛选未求解的文件
    unsolved_paths = []
    unsolved_smtlib = []
    solver_name = "MathSAT5"
    solver = MathSAT5Solver()
    timeout = 1200  # 设置超时时间为1200秒
    # 将JSON文件保存在脚本所在的目录下
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, f"{solver_name.lower()}_QF_LIA.json")
    
    # 获取CPU逻辑核心数
    cpu_cores = multiprocessing.cpu_count()
    max_workers = cpu_cores - 1
    
    # 1. 读取已有结果
    result_dict = load_result_dict(json_path)
    
    test_path = []
    directory = '/home/lz/Downloads/non-incremental_Hierarchy/non-incremental'
    test_path.append(directory)
    
    search_list = ['QF_LIA']
    
    # 遍历目录
    pattern = re.compile(r'\(set-info :status (\w+)\)')
    count = 0
    
    for directory in test_path:
        for search in search_list:
            for dirpath, dirnames, filenames in os.walk(os.path.join(directory, search)):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    
                    # 跳过特定文件
                    if any(skip in file_path for skip in ['starexec_description.txt', '52759_b3ecd2335fd16ec2eee2_9_UFDTBV', 'sll-optional-1.i_1']):
                        continue
                    
                    # 检查是否已存在结果（断点续传机制）
                    if file_path in result_dict:
                        continue
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            smtlib_str = file.read()
                        
                        match = pattern.search(smtlib_str)
                        if match:
                            status = match.group(1)
                            print(f"文件 {file_path} 状态: {status}")
                        else:
                            print(f"文件 {file_path} 未找到状态信息")
                            continue
                        
                        # 只处理状态不为unsat的文件
                        if status != 'unsat':
                            count += 1
                            unsolved_paths.append(file_path)
                            unsolved_smtlib.append(smtlib_str)
                    
                    except Exception as e:
                        print(f"读取文件 {file_path} 时出错: {e}")
                        continue
    
    # 3. 并发求解未求解的文件
    if unsolved_smtlib:
        print(f"开始使用MathSAT5求解 {len(unsolved_smtlib)} 个QF_LIA文件...")
        results = concurrent_solve(solver, unsolved_smtlib, timeout=timeout, max_workers=max_workers)
        
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
        print(f"MathSAT5新结果已保存到 {json_path}")
    else:
        print("所有文件均已存在结果，无需重复求解。")
    
    # 5. 输出统计信息
    print(f"MathSAT5总共处理了 {len(result_dict)} 个QF_LIA文件")

def get_CVC5_result_all():
    file_paths = []
    setup_logger()
    info_name = '/home/lz/sibyl_3/src/networks/info_dict_predictor.txt'
    info_dict = load_dictionary(info_name)
    file_paths = info_dict.keys()
    solver_name = "CVC5"
    solver = CVC5Solver()
    timeout = 1200  # 设置超时时间为1200秒
    json_path = f"{solver_name.lower()}_smtimer_results_predictor.json"

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

                smtlib_str = f.read()
                # 解析字符串
                try:
                    # 将JSON字符串转换为字典
                    dict_obj = json.loads(smtlib_str)
                    # print("转换后的字典：", dict_obj)
                except json.JSONDecodeError as e:
                    print("解析错误：", e)
                #
                if 'smt-comp' in path:
                    smtlib_str = dict_obj['smt_script']
                else:
                    smtlib_str = dict_obj['script']
                unsolved_smtlib.append(smtlib_str)

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
def get_MathSAT_result():
    file_paths = []
    setup_logger()
    info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1110_pre_SMTimer_llama3.1:70b_1200s_info_dict_rl.txt'
    info_dict = load_dictionary(info_name)
    file_paths = info_dict.keys()
    solver_name = "MathSAT5"
    solver = MathSAT5Solver()
    timeout = 1200  # 设置超时时间为1200秒
    json_path = f"{solver_name.lower()}_smtimer_results.json"

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

                smtlib_str = f.read()
                # 解析字符串
                try:
                    # 将JSON字符串转换为字典
                    dict_obj = json.loads(smtlib_str)
                    # print("转换后的字典：", dict_obj)
                except json.JSONDecodeError as e:
                    print("解析错误：", e)
                #
                if 'smt-comp' in path:
                    smtlib_str = dict_obj['smt_script']
                else:
                    smtlib_str = dict_obj['script']
                unsolved_smtlib.append(smtlib_str)

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

def get_OpenSMT_result():
    file_paths = []
    setup_logger()
    info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1110_pre_SMTimer_llama3.1:70b_1200s_info_dict_rl.txt'
    info_dict = load_dictionary(info_name)
    file_paths = info_dict.keys()
    solver_name = "OpenSMT"
    solver = OpenSMTSolver()
    timeout = 1200  # 设置超时时间为1200秒
    json_path = f"{solver_name.lower()}_smtimer_results.json"

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

                smtlib_str = f.read()
                # 解析字符串
                try:
                    # 将JSON字符串转换为字典
                    dict_obj = json.loads(smtlib_str)
                    # print("转换后的字典：", dict_obj)
                except json.JSONDecodeError as e:
                    print("解析错误：", e)
                #
                if 'smt-comp' in path:
                    smtlib_str = dict_obj['smt_script']
                else:
                    smtlib_str = dict_obj['script']
                unsolved_smtlib.append(smtlib_str)

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

def get_Yices_result():
    file_paths = []
    setup_logger()
    info_name = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1110_pre_SMTimer_llama3.1:70b_1200s_info_dict_rl.txt'
    info_dict = load_dictionary(info_name)
    file_paths = info_dict.keys()
    solver_name = "Yices"
    timeout = 1200  # 设置超时时间为1200秒
    json_path = f"{solver_name.lower()}_smtimer_results.json"

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

                smtlib_str = f.read()
                try:
                    dict_obj = json.loads(smtlib_str)
                except json.JSONDecodeError as e:
                    print("解析错误：", e)
                if 'smt-comp' in path:
                    smtlib_str = dict_obj['smt_script']
                else:
                    smtlib_str = dict_obj['script']
                unsolved_smtlib.append(smtlib_str)

    # 3. 并发求解未求解的文件（多进程）
    if unsolved_smtlib:
        results = concurrent_solve_process("yices", unsolved_smtlib, timeout=timeout, max_workers=3)
        for i, result in enumerate(results):
            model_dict = parse_model_to_dict(result["model"])
            result_dict[unsolved_paths[i]] = [
                result["result"],
                result["solve_time"],
                timeout,
                model_dict
            ]
        save_result_dict(json_path, result_dict)
        print(f"新结果已保存到 {json_path}")
    else:
        print("所有文件均已存在结果，无需重复求解。")

def test_z3solver_basic_sat():
    smtlib_str = '''
    (set-logic QF_UF)
    (declare-fun a () Bool)
    (assert a)
    '''
    solver = Z3Solver()
    result = solver.solve(smtlib_str, timeout=5)
    print("Test Z3Solver SAT:")
    print("  Result:", result.result)
    print("  Model:", result.model)
    assert result.result == 'sat', f"Expected 'sat', got {result.result}"
    assert 'a' in result.model, "Model should contain variable 'a'"

# 示例用法
if __name__ == "__main__":
    # 选择要使用的求解器，取消注释相应的行即可
    
    # === QF_NIA问题集 ===
    # get_CVC5_result_QF_NIA()
    # get_MathSAT_result_QF_NIA()
    
    # === QF_LIA问题集 ===
    get_CVC5_result_QF_LIA()       # 使用CVC5求解器处理QF_LIA问题
    get_MathSAT_result_QF_LIA()    # 使用MathSAT5求解器处理QF_LIA问题
    
    # === 其他求解器（用于不同的测试集）===
    # get_MathSAT_result()
    # get_OpenSMT_result()
    # get_Z3_result()
    # get_Yices_result()
    # 运行测试用例
    # test_z3solver_basic_sat()
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
