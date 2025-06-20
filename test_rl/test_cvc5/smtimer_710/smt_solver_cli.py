#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SMT Solver CLI - 命令行SMT求解器工具
支持多种SMT求解器：Z3, CVC5, MathSAT5, OpenSMT, Yices
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import multiprocessing
import tempfile
import time
import subprocess
from typing import List, Dict, Any, Optional

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from test_rl.test_script.search_test import load_dictionary
    from test_rl.test_script.utils import setup_logger, solve_and_measure_time, model_to_dict
    from loguru import logger
    from z3.z3 import parse_smt2_string, Solver as Z3_Solver, sat, unsat, unknown
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在项目根目录运行此脚本，或设置正确的PYTHONPATH")
    sys.exit(1)


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


def get_solver(solver_name: str) -> Solver:
    """根据求解器名称获取对应的求解器实例"""
    solver_map = {
        'z3': Z3Solver,
        'cvc5': CVC5Solver,
        'mathsat5': MathSAT5Solver,
        'opensmt': OpenSMTSolver,
        'yices': YicesSolver
    }
    
    solver_class = solver_map.get(solver_name.lower())
    if not solver_class:
        raise ValueError(f"不支持的求解器: {solver_name}. 支持的求解器: {list(solver_map.keys())}")
    
    return solver_class()


def concurrent_solve(solver, smtlib_list, timeout=5, max_workers=4):
    """并发求解多个SMT问题"""
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
    """使用指定求解器求解单个SMT问题"""
    solver = get_solver(solver_name)
    result = solver.solve(smtlib_str, timeout)
    return {
        "result": result.result,
        "solve_time": result.solve_time,
        "timeout": timeout,
        "model": result.model
    }


def concurrent_solve_process(solver_name, smtlib_list, timeout=5, max_workers=4):
    """多进程并发求解"""
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
    """将求解器输出的模型字符串转为字典"""
    model = {}
    if not model_str:
        return model
    lines = model_str.splitlines()
    for line in lines:
        if line.strip().startswith('(define-fun'):
            parts = line.strip().split()
            if len(parts) >= 4:
                var = parts[1]
                for v in parts[3:]:
                    if v not in ('(', ')'):
                        model[var] = v
                        break
    return model


def load_result_dict(json_path):
    """加载已有的结果字典"""
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_result_dict(json_path, result_dict):
    """保存结果字典到JSON文件"""
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)


def process_z3_file(args):
    """处理单个Z3文件的函数（用于多进程）"""
    file_path, timeout = args
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
        solver = Z3_Solver()
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


def solve_batch_files(info_file_path: str, solver_name: str, timeout: int = 1200, 
                     max_workers: int = 4, output_file: str = None, 
                     path_replace: Dict[str, str] = None):
    """
    批量求解文件
    
    Args:
        info_file_path: 包含文件路径信息的文件
        solver_name: 求解器名称
        timeout: 超时时间（秒）
        max_workers: 最大并发数
        output_file: 输出结果文件名
        path_replace: 路径替换映射
    """
    setup_logger()
    
    # 加载文件路径信息
    info_dict = load_dictionary(info_file_path)
    file_paths = list(info_dict.keys())
    
    # 路径替换
    if path_replace:
        for i, path in enumerate(file_paths):
            for old_path, new_path in path_replace.items():
                if old_path in path:
                    file_paths[i] = path.replace(old_path, new_path)
                    break
    
    # 设置输出文件名
    if not output_file:
        output_file = f"{solver_name.lower()}_smtimer_results.json"
    
    logger.info(f"{solver_name}批量求解任务启动，待处理文件数: {len(file_paths)}")
    
    # 读取已有结果
    result_dict = load_result_dict(output_file)
    
    # 筛选未求解的文件
    unsolved_paths = [p for p in file_paths if p not in result_dict]
    logger.info(f"未求解文件数: {len(unsolved_paths)}")
    
    if not unsolved_paths:
        logger.info("所有文件均已存在结果，无需重复求解。")
        print("所有文件均已存在结果，无需重复求解。")
        return
    
    # 根据求解器选择不同的求解策略
    if solver_name.lower() == 'z3':
        # Z3使用多进程
        tasks = [(path, timeout) for path in unsolved_paths]
        with multiprocessing.Pool(processes=max_workers) as pool:
            for file_path, result_list in pool.imap_unordered(process_z3_file, tasks):
                result_dict[file_path] = result_list
                logger.info(f"主进程收集结果: {file_path}, 状态: {result_list[0]}")
                print(f"已完成: {file_path}, 结果: {result_list[0]}")
    else:
        # 其他求解器使用多线程
        solver = get_solver(solver_name)
        unsolved_smtlib = []
        
        for path in unsolved_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    smtlib_str = f.read()
                try:
                    dict_obj = json.loads(smtlib_str)
                    if 'smt-comp' in path:
                        smtlib_str = dict_obj['smt_script']
                    else:
                        smtlib_str = dict_obj['script']
                except json.JSONDecodeError:
                    pass
                unsolved_smtlib.append(smtlib_str)
            except Exception as e:
                logger.error(f"读取文件失败: {path}, 错误: {e}")
                continue
        
        if unsolved_smtlib:
            results = concurrent_solve(solver, unsolved_smtlib, timeout=timeout, max_workers=max_workers)
            for i, result in enumerate(results):
                model_dict = parse_model_to_dict(result.model)
                result_dict[unsolved_paths[i]] = [
                    result.result,
                    result.solve_time,
                    timeout,
                    model_dict
                ]
    
    # 保存结果
    save_result_dict(output_file, result_dict)
    logger.info(f"全部结果已保存到 {output_file}")
    print(f"新结果已保存到 {output_file}")


def solve_single_file(file_path: str, solver_name: str, timeout: int = 5):
    """求解单个文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            smtlib_str = f.read()
        
        # 尝试解析JSON格式
        try:
            dict_obj = json.loads(smtlib_str)
            if 'smt-comp' in file_path:
                smtlib_str = dict_obj['smt_script']
            else:
                smtlib_str = dict_obj['script']
        except json.JSONDecodeError:
            pass  # 不是JSON格式，直接使用原始内容
        
        solver = get_solver(solver_name)
        result = solver.solve(smtlib_str, timeout)
        
        print(f"文件: {file_path}")
        print(f"求解器: {solver_name}")
        print(f"结果: {result.result}")
        print(f"用时: {result.solve_time:.2f}秒")
        print(f"模型: {result.model}")
        
        return result
        
    except Exception as e:
        print(f"求解失败: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='SMT求解器命令行工具')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 批量求解命令
    batch_parser = subparsers.add_parser('batch', help='批量求解文件')
    batch_parser.add_argument('--info-file', required=True, help='包含文件路径信息的文件')
    batch_parser.add_argument('--solver', required=True, choices=['z3', 'cvc5', 'mathsat5', 'opensmt', 'yices'], 
                             help='求解器名称')
    batch_parser.add_argument('--timeout', type=int, default=1200, help='超时时间（秒）')
    batch_parser.add_argument('--max-workers', type=int, default=4, help='最大并发数')
    batch_parser.add_argument('--output', help='输出结果文件名')
    batch_parser.add_argument('--path-replace', nargs=2, action='append', 
                             help='路径替换，格式: 旧路径 新路径')
    
    # 单文件求解命令
    single_parser = subparsers.add_parser('single', help='求解单个文件')
    single_parser.add_argument('--file', required=True, help='要求解的文件路径')
    single_parser.add_argument('--solver', required=True, choices=['z3', 'cvc5', 'mathsat5', 'opensmt', 'yices'], 
                              help='求解器名称')
    single_parser.add_argument('--timeout', type=int, default=5, help='超时时间（秒）')
    
    # 测试命令
    test_parser = subparsers.add_parser('test', help='运行测试用例')
    
    args = parser.parse_args()
    
    if args.command == 'batch':
        path_replace = None
        if args.path_replace:
            path_replace = dict(args.path_replace)
        
        solve_batch_files(
            info_file_path=args.info_file,
            solver_name=args.solver,
            timeout=args.timeout,
            max_workers=args.max_workers,
            output_file=args.output,
            path_replace=path_replace
        )
    
    elif args.command == 'single':
        solve_single_file(
            file_path=args.file,
            solver_name=args.solver,
            timeout=args.timeout
        )
    
    elif args.command == 'test':
        # 运行基本测试
        smtlib_str = '''
        (set-logic QF_UF)
        (declare-fun a () Bool)
        (assert a)
        '''
        solver = Z3Solver()
        result = solver.solve(smtlib_str, timeout=5)
        print("测试Z3求解器SAT:")
        print(f"  结果: {result.result}")
        print(f"  模型: {result.model}")
        assert result.result == 'sat', f"期望'sat'，得到{result.result}"
        assert 'a' in result.model, "模型应包含变量'a'"
        print("测试通过！")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 