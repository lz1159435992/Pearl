#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import subprocess
from z3.z3 import parse_smt2_string, Solver as Z3_Solver, sat, unsat, unknown
import multiprocessing
from loguru import logger
import sys

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
            logger.info(f"运行命令: {' '.join(cmd)}")
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
            logger.info(f"运行命令: {' '.join(cmd)}")
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
            logger.info(f"运行命令: {' '.join(cmd)}")
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
            logger.info(f"运行命令: {' '.join(cmd)}")
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
    elif solver_name.lower() == "cvc5":
        solver = CVC5Solver()
    elif solver_name.lower() == "mathsat5":
        solver = MathSAT5Solver()
    elif solver_name.lower() == "opensmt":
        solver = OpenSMTSolver()
    elif solver_name.lower() == "yices":
        solver = YicesSolver()
    else:
        raise ValueError(f"未知求解器: {solver_name}")
    result = solver.solve(smtlib_str, timeout)
    return {
        "result": result.result,
        "solve_time": result.solve_time,
        "timeout": timeout,
        "model": result.model
    }

def concurrent_solve_process(solver_name, smtlib_list, timeout=5, max_workers=4):
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
    """尝试将求解器输出的模型字符串转为字典"""
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

def load_dictionary(file_path):
    """加载字典"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_result_dict(json_path):
    """加载结果字典，如果文件不存在则返回空字典"""
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_result_dict(json_path, result_dict):
    """保存结果字典到文件"""
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)

def setup_logger():
    """设置日志记录器"""
    logger.remove()
    logger.add("solver_script.log", rotation="10 MB")
    logger.add(sys.stderr, level="INFO")  # 添加控制台输出
    return logger

def get_solver_instance(solver_name):
    """根据求解器名称获取求解器实例"""
    solvers = {
        "cvc5": CVC5Solver(),
        "z3": Z3Solver(),
        "mathsat5": MathSAT5Solver(),
        "opensmt": OpenSMTSolver(),
        "yices": YicesSolver(),
    }
    
    solver_name = solver_name.lower()
    if solver_name not in solvers:
        raise ValueError(f"不支持的求解器: {solver_name}")
    
    return solvers[solver_name]

def solve_with_solver(solver_name, info_dict_path, output_json, timeout=1200, max_workers=4):
    """使用指定求解器解决SMT问题"""
    setup_logger()
    
    logger.info(f"使用 {solver_name} 求解器开始批量求解")
    
    # 加载信息字典，获取文件路径
    info_dict = load_dictionary(info_dict_path)
    file_paths = list(info_dict.keys())
    
    # 确定输出文件路径
    if not output_json:
        output_json = f"{solver_name.lower()}_solver_results.json"
    
    # 加载已有结果
    result_dict = load_result_dict(output_json)
    logger.info(f"已加载已有结果，共 {len(result_dict)} 条")
    
    # 筛选未求解的文件
    unsolved_paths = []
    unsolved_smtlib = []
    
    for path in file_paths:
        if path in result_dict:
            logger.debug(f"{solver_name}: {path} 已存在，跳过。")
        else:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    smtlib_str = f.read()
                    
                    # 尝试解析JSON格式
                    try:
                        dict_obj = json.loads(smtlib_str)
                        if 'smt-comp' in path:
                            smtlib_str = dict_obj['smt_script']
                        else:
                            smtlib_str = dict_obj['script']
                    except json.JSONDecodeError:
                        # 不是JSON格式，直接使用原始内容
                        pass
                    
                    unsolved_paths.append(path)
                    unsolved_smtlib.append(smtlib_str)
                    logger.info(f"添加未求解文件: {path}")
            except Exception as e:
                logger.error(f"读取文件 {path} 时出错: {e}")
    
    logger.info(f"共找到 {len(unsolved_paths)} 个未求解文件")
    
    # 根据求解器选择不同的求解方法
    if solver_name.lower() == "yices":
        # Yices使用多进程
        if unsolved_smtlib:
            logger.info(f"使用多进程并行求解 {len(unsolved_smtlib)} 个文件")
            results = concurrent_solve_process(solver_name, unsolved_smtlib, timeout=timeout, max_workers=max_workers)
            
            for i, result in enumerate(results):
                model_dict = parse_model_to_dict(result["model"])
                result_dict[unsolved_paths[i]] = [
                    result["result"],
                    result["solve_time"],
                    timeout,
                    model_dict
                ]
                logger.info(f"完成 {unsolved_paths[i]}: {result['result']}, 用时: {result['solve_time']:.2f}s")
            
            save_result_dict(output_json, result_dict)
            logger.info(f"新结果已保存到 {output_json}")
        else:
            logger.info("所有文件均已存在结果，无需重复求解。")
    else:
        # 其他求解器使用多线程
        if unsolved_smtlib:
            solver = get_solver_instance(solver_name)
            logger.info(f"使用多线程并行求解 {len(unsolved_smtlib)} 个文件")
            results = concurrent_solve(solver, unsolved_smtlib, timeout=timeout, max_workers=max_workers)
            
            for i, result in enumerate(results):
                model_dict = parse_model_to_dict(result.model)
                result_dict[unsolved_paths[i]] = [
                    result.result,
                    result.solve_time,
                    timeout,
                    model_dict
                ]
                logger.info(f"完成 {unsolved_paths[i]}: {result.result}, 用时: {result.solve_time:.2f}s")
            
            save_result_dict(output_json, result_dict)
            logger.info(f"新结果已保存到 {output_json}")
        else:
            logger.info("所有文件均已存在结果，无需重复求解。")
    
    return result_dict

def process_z3_file(args):
    """Z3特殊处理函数，用于多进程"""
    file_path, timeout = args
    try:
        logger.info(f"[工作进程] 开始处理: {file_path}")
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
        
        solver = Z3Solver()
        result = solver.solve(smtlib_str, timeout)
        model_dict = parse_model_to_dict(result.model)
        logger.info(f"[工作进程] 完成: {file_path}, 结果: {result.result}, 用时: {result.solve_time:.2f}s")
        return (file_path, [result.result, result.solve_time, timeout, model_dict])
    except Exception as e:
        logger.info(f"[工作进程] 处理异常: {file_path}, 错误: {e}")
        return (file_path, ['error', 0, timeout, str(e)])

def main():
    parser = argparse.ArgumentParser(description="SMT求解器批量求解工具")
    parser.add_argument("--solver", type=str, required=False, choices=["cvc5", "z3", "mathsat5", "opensmt", "yices"],
                        default="mathsat5",
                      help="选择要使用的求解器 (默认: mathsat5)")
    parser.add_argument("--info-dict", type=str, required=False,
                        default="/home/lz/sibyl_3/src/networks/info_dict_predictor.txt",
                      help="包含SMT问题文件路径的信息字典文件")
    parser.add_argument("--output", type=str,
                        default="mathsat5_smtimer_results_predictor.txt",
                      help="保存结果的JSON文件路径 (默认: <solver_name>_solver_results.json)")
    parser.add_argument("--timeout", type=int, default=1200,
                      help="每个问题的超时时间(秒) (默认: 1200)")
    parser.add_argument("--workers", type=int, default=4,
                      help="并行处理的工作进程/线程数 (默认: 4)")
    
    args = parser.parse_args()
    
    try:
        result_dict = solve_with_solver(
            args.solver, 
            args.info_dict, 
            args.output, 
            timeout=args.timeout, 
            max_workers=args.workers
        )
        
        print(f"\n求解完成. 共处理 {len(result_dict)} 个文件.")
        print(f"结果已保存到 {args.output or f'{args.solver.lower()}_solver_results.json'}")
        
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()