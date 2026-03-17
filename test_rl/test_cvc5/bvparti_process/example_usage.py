#!/usr/bin/env python3
"""
BVParti预测器使用示例
演示如何使用BVParti预测器处理SMT约束求解问题
"""

import os
import sys
import json
import tempfile
from loguru import logger

# 添加路径
sys.path.append('/home/lz/PycharmProjects/Pearl')

def test_bvparti_solver_only():
    """仅测试BVParti求解器功能"""
    print("\n=== 测试BVParti求解器 ===")

    try:
        from test_rl.test_cvc5.bvparti_process.run_bvparti_predictor import BVPartiSolver

        solver = BVPartiSolver()

        # 简单的SAT问题
        smt_sat = """(set-logic QF_BV)
(declare-fun x () (_ BitVec 8))
(assert (= x (_ bv42 8)))
(check-sat)
(exit)"""

        print("测试SAT问题...")
        result = solver.solve(smt_sat, timeout=10)
        print(f"结果: {result.result}, 耗时: {result.solve_time:.3f}秒")

        # 简单的UNSAT问题
        smt_unsat = """(set-logic QF_BV)
(declare-fun x () (_ BitVec 8))
(assert (= x (_ bv42 8)))
(assert (= x (_ bv43 8)))
(check-sat)
(exit)"""

        print("测试UNSAT问题...")
        result = solver.solve(smt_unsat, timeout=10)
        print(f"结果: {result.result}, 耗时: {result.solve_time:.3f}秒")

        return True

    except Exception as e:
        print(f"测试求解器时出错: {e}")
        return False

def show_command_examples():
    """显示命令行使用示例"""
    print("\n=== 命令行使用示例 ===")

    examples = [
        {
            "name": "基本用法（使用BVParti）",
            "command": """python test_rl/test_cvc5/bvparti_process/run_bvparti_predictor.py \\
    --solver bvparti \\
    --result_dict_path /path/to/SMTimer_z3_result_rl.json \\
    --info_dict_path output_info_dict.txt \\
    --timeout 1200"""
        },
        {
            "name": "使用Z3求解器",
            "command": """python test_rl/test_cvc5/bvparti_process/run_bvparti_predictor.py \\
    --solver z3 \\
    --result_dict_path /path/to/cvc5_smtimer_results.json \\
    --info_dict_path z3_results.txt \\
    --timeout 600"""
        },
        {
            "name": "自定义LLM设置",
            "command": """python test_rl/test_cvc5/bvparti_process/run_bvparti_predictor.py \\
    --solver bvparti \\
    --llm_host http://localhost:11434 \\
    --llm_model llama3.1:8b \\
    --num_episodes 5 \\
    --timeout 1800"""
        },
        {
            "name": "快速测试模式",
            "command": """python test_rl/test_cvc5/bvparti_process/run_bvparti_predictor.py \\
    --solver bvparti \\
    --time_threshold 100 \\
    --timeout 300 \\
    --num_episodes 1"""
        }
    ]

    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['name']}:")
        print(example['command'])

def main():
    """主函数"""
    print("BVParti预测器使用示例")
    print("=" * 50)

    # 运行各种演示
    tests = [
        ("BVParti求解器测试", test_bvparti_solver_only),
    ]

    for test_name, test_func in tests:
        try:
            success = test_func()
            status = "✓ 成功" if success else "✗ 失败"
            print(f"\n{test_name}: {status}")
        except Exception as e:
            print(f"\n{test_name}: ✗ 异常 - {e}")

    # 显示命令行示例
    show_command_examples()

    print("\n" + "=" * 50)
    print("演示完成！")
    print("\n要运行完整的预测器，请使用以下命令:")
    print("python test_rl/test_cvc5/bvparti_process/run_bvparti_predictor.py --help")
    print("\n更多信息请参考: README_BVPARTI_PREDICTOR.md")

if __name__ == "__main__":
    main()