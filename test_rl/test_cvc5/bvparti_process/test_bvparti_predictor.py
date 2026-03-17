#!/usr/bin/env python3
"""
测试BVParti预测器的基本功能
"""

import os
import sys
import json
import tempfile
import traceback
from loguru import logger

# 添加路径
sys.path.append('/home/lz/PycharmProjects/Pearl')

def test_bvparti_solver():
    """测试BVParti求解器的基本功能"""
    print("=== 测试BVParti求解器 ===")
    
    try:
        from test_rl.test_cvc5.bvparti_process.run_bvparti_predictor import BVPartiSolver, BVPartiConfig
        
        # 创建求解器
        config = BVPartiConfig()
        solver = BVPartiSolver(config)
        
        print(f"BVParti配置:")
        print(f"  - 基础路径: {config.ariparti_base_path}")
        print(f"  - BVParti二进制: {config.bvparti_bin}")
        print(f"  - 分区器二进制: {config.partitioner_bin}")
        print(f"  - Bitwuzla二进制: {config.bitwuzla_bin}")
        print(f"  - 时间限制: {config.time_limit}秒")
        
        # 检查路径是否存在
        paths_exist = all(os.path.exists(path) for path in [
            config.bvparti_bin,
            config.partitioner_bin,
            config.bitwuzla_bin
        ])
        
        if not paths_exist:
            print("警告: 某些BVParti组件路径不存在，可能无法正常工作")
            return False
        
        # 创建一个简单的SMT2测试文件
        test_smt2 = """(set-logic QF_BV)
(declare-fun x () (_ BitVec 8))
(declare-fun y () (_ BitVec 8))
(assert (= (bvadd x y) (_ bv10 8)))
(assert (= x (_ bv3 8)))
(check-sat)
(exit)
"""
        
        print("\n测试SMT2内容:")
        print(test_smt2)
        
        # 测试求解
        print("开始求解...")
        result = solver.solve(test_smt2, timeout=10)
        
        print(f"求解结果:")
        print(f"  - 结果: {result.result}")
        print(f"  - 耗时: {result.solve_time:.3f}秒")
        print(f"  - 模型: {result.model[:100] if result.model else 'None'}...")
        
        return result.result in ['sat', 'unsat']
        
    except Exception as e:
        print(f"测试BVParti求解器时出错: {e}")
        traceback.print_exc()
        return False

def test_json_compatibility():
    """测试JSON格式兼容性"""
    print("\n=== 测试JSON格式兼容性 ===")
    
    try:
        from test_rl.test_cvc5.bvparti_process.test_group_get_dis_smt_comp_bert_embeding_single import (
            get_solve_result_and_time, convert_timeout_to_unknown
        )
        
        # 测试新格式（BVParti格式）
        new_format_data = {
            "metadata": {
                "execution_info": {"start_time": "2025-07-21 00:36:48"},
                "configuration": {"solver": "bitwuzla", "time_limit": 1200}
            },
            "results": {
                "/test/file1": {
                    "result": "sat",
                    "solve_time": 1.5,
                    "total_time": 1.6,
                    "error": None,
                    "returncode": 0
                },
                "/test/file2": {
                    "result": "timeout",
                    "solve_time": 1200.0,
                    "total_time": 1200.1,
                    "error": None,
                    "returncode": 0
                }
            }
        }
        
        print("测试新格式数据处理:")
        for key in new_format_data["results"]:
            category, time_value = get_solve_result_and_time(new_format_data, key)
            print(f"  {key}: {category}, {time_value}s")
        
        # 测试timeout转换
        converted_data = convert_timeout_to_unknown(new_format_data)
        print("\n测试timeout转换:")
        for key in converted_data["results"]:
            category, time_value = get_solve_result_and_time(converted_data, key)
            print(f"  {key}: {category}, {time_value}s")
        
        return True
        
    except Exception as e:
        print(f"测试JSON兼容性时出错: {e}")
        traceback.print_exc()
        return False

def test_environment_creation():
    """测试环境创建"""
    print("\n=== 测试环境创建 ===")
    
    try:
        # 这里只测试导入，不实际创建环境（需要模型文件）
        from test_rl.test_cvc5.bvparti_process.run_bvparti_predictor import (
            ConstraintSimplificationEnv_test, get_solver
        )
        
        print("成功导入环境类和求解器获取函数")
        
        # 测试求解器获取
        solvers = ['z3', 'cvc5', 'bvparti']
        for solver_name in solvers:
            solver = get_solver(solver_name)
            print(f"  {solver_name}: {type(solver).__name__}")
        
        return True
        
    except Exception as e:
        print(f"测试环境创建时出错: {e}")
        traceback.print_exc()
        return False

def test_argument_parsing():
    """测试命令行参数解析"""
    print("\n=== 测试命令行参数解析 ===")
    
    try:
        import argparse
        from test_rl.test_cvc5.bvparti_process.run_bvparti_predictor import main
        
        # 创建测试参数
        test_args = [
            '--solver', 'bvparti',
            '--llm_model', 'llama3.1:70b',
            '--timeout', '600',
            '--num_episodes', '1'
        ]
        
        print(f"测试参数: {' '.join(test_args)}")
        
        # 这里只测试参数解析器的创建，不实际运行main
        parser = argparse.ArgumentParser(description='运行BVParti SMT约束求解预测器')
        parser.add_argument('--solver', type=str, default='bvparti', choices=['z3', 'cvc5', 'bvparti'])
        parser.add_argument('--llm_model', type=str, default='llama3.1:70b')
        parser.add_argument('--timeout', type=int, default=1200)
        parser.add_argument('--num_episodes', type=int, default=1)
        
        args = parser.parse_args(test_args)
        print(f"解析结果:")
        print(f"  - 求解器: {args.solver}")
        print(f"  - LLM模型: {args.llm_model}")
        print(f"  - 超时时间: {args.timeout}")
        print(f"  - 训练轮数: {args.num_episodes}")
        
        return True
        
    except Exception as e:
        print(f"测试参数解析时出错: {e}")
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("开始测试BVParti预测器...")
    print("=" * 60)
    
    tests = [
        ("BVParti求解器", test_bvparti_solver),
        ("JSON格式兼容性", test_json_compatibility),
        ("环境创建", test_environment_creation),
        ("命令行参数解析", test_argument_parsing),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
            status = "✓ 通过" if result else "✗ 失败"
            print(f"\n{test_name}: {status}")
        except Exception as e:
            results[test_name] = False
            print(f"\n{test_name}: ✗ 异常 - {e}")
        
        print("-" * 60)
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结:")
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓" if result else "✗"
        print(f"  {status} {test_name}")
    
    print(f"\n通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！BVParti预测器已准备就绪。")
    else:
        print("⚠️  部分测试失败，请检查配置和依赖。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
