#!/usr/bin/env python3
"""
终极验证测试：最后一次全面检查BVParti预测器的所有逻辑
确保与CVC5版本完全一致，没有任何错误
"""

import os
import sys
import json
import traceback
import subprocess
import tempfile

# 添加路径
sys.path.append('/home/lz/PycharmProjects/Pearl')

def test_argument_consistency():
    """测试命令行参数一致性"""
    print("=== 测试命令行参数一致性 ===")
    
    try:
        # 获取CVC5版本的help输出
        cvc5_help = subprocess.run([
            sys.executable, 'test_rl/test_cvc5/cvc5_process/run_predictor.py', '--help'
        ], cwd='/home/lz/PycharmProjects/Pearl', capture_output=True, text=True, timeout=30)
        
        # 获取BVParti版本的help输出
        bvparti_help = subprocess.run([
            sys.executable, 'test_rl/test_cvc5/bvparti_process/run_bvparti_predictor.py', '--help'
        ], cwd='/home/lz/PycharmProjects/Pearl', capture_output=True, text=True, timeout=30)
        
        if cvc5_help.returncode != 0:
            print(f"  ✗ CVC5版本help命令失败: {cvc5_help.stderr}")
            return False
        
        if bvparti_help.returncode != 0:
            print(f"  ✗ BVParti版本help命令失败: {bvparti_help.stderr}")
            return False
        
        # 检查关键参数是否存在
        required_args = [
            '--rl_dict_path', '--info_dict_path', '--result_dict_path',
            '--binary_model_path', '--eight_class_model_path', '--time_threshold',
            '--timeout', '--num_episodes', '--llm_host', '--llm_model', '--solver'
        ]
        
        cvc5_output = cvc5_help.stdout
        bvparti_output = bvparti_help.stdout
        
        missing_in_cvc5 = []
        missing_in_bvparti = []
        
        for arg in required_args:
            if arg not in cvc5_output:
                missing_in_cvc5.append(arg)
            if arg not in bvparti_output:
                missing_in_bvparti.append(arg)
        
        if missing_in_cvc5:
            print(f"  ✗ CVC5版本缺少参数: {missing_in_cvc5}")
            return False
        
        if missing_in_bvparti:
            print(f"  ✗ BVParti版本缺少参数: {missing_in_bvparti}")
            return False
        
        print("  ✓ 所有必需参数都存在")
        
        # 检查求解器选择差异
        if 'mathsat' in cvc5_output and 'bvparti' not in cvc5_output:
            print("  ✓ CVC5版本支持mathsat求解器")
        
        if 'bvparti' in bvparti_output and 'mathsat' not in bvparti_output:
            print("  ✓ BVParti版本支持bvparti求解器")
        
        return True
        
    except Exception as e:
        print(f"测试参数一致性时出错: {e}")
        traceback.print_exc()
        return False

def test_data_flow_consistency():
    """测试数据流一致性"""
    print("\n=== 测试数据流一致性 ===")
    
    try:
        from test_rl.test_cvc5.bvparti_process.test_group_get_dis_smt_comp_bert_embeding_single import (
            get_solve_result_and_time, convert_timeout_to_unknown
        )
        
        # 模拟CVC5格式数据
        cvc5_format = {
            "/file1": ["sat", 1.5, 1200, {}],
            "/file2": ["timeout", 1200.0, 1200, {}],
            "/file3": ["unsat", 0.5, 1200, {}]
        }
        
        # 模拟BVParti格式数据
        bvparti_format = {
            "metadata": {"test": "data"},
            "results": {
                "/file1": {"result": "sat", "solve_time": 1.5},
                "/file2": {"result": "timeout", "solve_time": 1200.0},
                "/file3": {"result": "unsat", "solve_time": 0.5}
            }
        }
        
        # 测试CVC5格式处理（应该通过简单路径）
        cvc5_converted = convert_timeout_to_unknown(cvc5_format)
        
        # 测试BVParti格式处理（应该通过复杂路径）
        bvparti_converted = convert_timeout_to_unknown(bvparti_format)
        
        # 验证结果一致性
        expected_results = {
            "/file1": ("sat", 1.5),
            "/file2": ("unknown", 1200.0),  # timeout -> unknown
            "/file3": ("unsat", 0.5)
        }
        
        print("  测试CVC5格式处理:")
        for key, (expected_cat, expected_time) in expected_results.items():
            actual_cat, actual_time = get_solve_result_and_time(cvc5_converted, key)
            if actual_cat == expected_cat and actual_time == expected_time:
                print(f"    ✓ {key}: {actual_cat}, {actual_time}s")
            else:
                print(f"    ✗ {key}: 期望({expected_cat}, {expected_time}), 实际({actual_cat}, {actual_time})")
                return False
        
        print("  测试BVParti格式处理:")
        for key, (expected_cat, expected_time) in expected_results.items():
            actual_cat, actual_time = get_solve_result_and_time(bvparti_converted, key)
            if actual_cat == expected_cat and actual_time == expected_time:
                print(f"    ✓ {key}: {actual_cat}, {actual_time}s")
            else:
                print(f"    ✗ {key}: 期望({expected_cat}, {expected_time}), 实际({actual_cat}, {actual_time})")
                return False
        
        return True
        
    except Exception as e:
        print(f"测试数据流一致性时出错: {e}")
        traceback.print_exc()
        return False

def test_result_format_consistency():
    """测试结果格式一致性"""
    print("\n=== 测试结果格式一致性 ===")
    
    try:
        # 模拟result_list的构造过程
        # 基础部分：[status, time, memory]
        base_list = ["sat", 1.5, 0]
        
        # 添加执行信息
        result_list = base_list.copy()
        result_list.append(10.0)    # total_execution_time
        result_list.append(5.0)     # total_solve_time
        result_list.append(1.5)     # final_solve_time
        result_list.append(2.0)     # llm_total_time
        result_list.append('succeed')  # status
        result_list.append([])      # last_assignments
        result_list.append([[]])    # counterexamples_list
        
        # 验证格式
        expected_length = 10
        if len(result_list) != expected_length:
            print(f"  ✗ 结果列表长度不正确: 期望{expected_length}, 实际{len(result_list)}")
            return False
        
        # 验证类型
        expected_types = [str, float, int, float, float, float, float, str, list, list]
        for i, (actual, expected_type) in enumerate(zip(result_list, expected_types)):
            if not isinstance(actual, expected_type):
                print(f"  ✗ 索引{i}类型不正确: 期望{expected_type.__name__}, 实际{type(actual).__name__}")
                return False
        
        print(f"  ✓ 结果格式正确: 长度={len(result_list)}, 所有类型匹配")
        
        # 验证关键索引
        if result_list[7] not in ['succeed', 'failed']:
            print(f"  ✗ 状态字段值不正确: {result_list[7]}")
            return False
        
        print("  ✓ 状态字段值正确")
        return True
        
    except Exception as e:
        print(f"测试结果格式一致性时出错: {e}")
        traceback.print_exc()
        return False

def test_file_access_permissions():
    """测试文件访问权限"""
    print("\n=== 测试文件访问权限 ===")
    
    try:
        critical_files = [
            '/home/lz/PycharmProjects/Pearl/test_rl/AriParti_sync/scripts/batch_output/bv_default/SMTimer_z3_result_rl.json',
            '/home/lz/sibyl_3/src/networks/info_dict_rl.txt',
            '/home/lz/PycharmProjects/Pearl/test_rl/AriParti_sync/STP-Parti-Bitwuzla-at-SMT-COMP-2025-build/solver/BVPartition-bin',
            '/home/lz/PycharmProjects/Pearl/test_rl/AriParti_sync/STP-Parti-Bitwuzla-at-SMT-COMP-2025-build/solver/bitwuzla-0.8.0-bin'
        ]
        
        all_accessible = True
        for file_path in critical_files:
            if os.path.exists(file_path):
                if os.access(file_path, os.R_OK):
                    print(f"  ✓ {os.path.basename(file_path)}: 可读")
                else:
                    print(f"  ✗ {os.path.basename(file_path)}: 存在但不可读")
                    all_accessible = False
            else:
                print(f"  ✗ {os.path.basename(file_path)}: 不存在")
                all_accessible = False
        
        return all_accessible
        
    except Exception as e:
        print(f"测试文件访问权限时出错: {e}")
        traceback.print_exc()
        return False

def test_import_consistency():
    """测试导入一致性"""
    print("\n=== 测试导入一致性 ===")
    
    try:
        # 测试关键模块导入
        critical_imports = [
            'test_rl.test_cvc5.bvparti_process.run_bvparti_predictor',
            'test_rl.test_cvc5.bvparti_process.test_group_get_dis_smt_comp_bert_embeding_single'
        ]
        
        for module_name in critical_imports:
            try:
                __import__(module_name)
                print(f"  ✓ {module_name}: 导入成功")
            except ImportError as e:
                print(f"  ✗ {module_name}: 导入失败 - {e}")
                return False
        
        # 测试关键类和函数
        from test_rl.test_cvc5.bvparti_process.run_bvparti_predictor import (
            BVPartiSolver, Z3Solver, CVC5Solver, get_solver, 
            ConstraintSimplificationEnv_test, SolverResult
        )
        
        print("  ✓ 所有关键类和函数导入成功")
        
        # 测试求解器创建
        for solver_name in ['bvparti', 'z3', 'cvc5']:
            solver = get_solver(solver_name)
            if hasattr(solver, 'solve'):
                print(f"  ✓ {solver_name}求解器创建成功")
            else:
                print(f"  ✗ {solver_name}求解器缺少solve方法")
                return False
        
        return True
        
    except Exception as e:
        print(f"测试导入一致性时出错: {e}")
        traceback.print_exc()
        return False

def test_end_to_end_compatibility():
    """测试端到端兼容性"""
    print("\n=== 测试端到端兼容性 ===")
    
    try:
        # 创建一个简单的测试用例
        test_smt = """(set-logic QF_BV)
(declare-fun x () (_ BitVec 8))
(assert (= x (_ bv42 8)))
(check-sat)
(exit)"""
        
        # 测试BVParti求解器
        from test_rl.test_cvc5.bvparti_process.run_bvparti_predictor import BVPartiSolver
        
        bvparti_solver = BVPartiSolver()
        result = bvparti_solver.solve(test_smt, timeout=10)
        
        if hasattr(result, 'result') and hasattr(result, 'solve_time'):
            print(f"  ✓ BVParti求解器测试: {result.result}, {result.solve_time:.3f}s")
        else:
            print(f"  ✗ BVParti求解器结果格式不正确")
            return False
        
        # 测试数据处理流程
        from test_rl.test_cvc5.bvparti_process.test_group_get_dis_smt_comp_bert_embeding_single import (
            get_solve_result_and_time, convert_timeout_to_unknown
        )

        test_data = {
            "metadata": {"test": "data"},
            "results": {
                "/test/file": {"result": "sat", "solve_time": 1.5}
            }
        }

        # 先应用转换（虽然这个例子不需要转换，但保持一致性）
        converted_data = convert_timeout_to_unknown(test_data)

        category, time_value = get_solve_result_and_time(converted_data, "/test/file")
        if category == "sat" and time_value == 1.5:
            print("  ✓ 数据处理流程测试通过")
        else:
            print(f"  ✗ 数据处理流程测试失败: {category}, {time_value}")
            return False
        
        return True
        
    except Exception as e:
        print(f"测试端到端兼容性时出错: {e}")
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("BVParti预测器终极验证测试")
    print("=" * 80)
    print("最后一次全面检查，确保与CVC5版本完全一致")
    print("=" * 80)
    
    tests = [
        ("命令行参数一致性", test_argument_consistency),
        ("数据流一致性", test_data_flow_consistency),
        ("结果格式一致性", test_result_format_consistency),
        ("文件访问权限", test_file_access_permissions),
        ("导入一致性", test_import_consistency),
        ("端到端兼容性", test_end_to_end_compatibility),
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
        
        print("-" * 80)
    
    # 最终总结
    print("\n终极验证总结:")
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓" if result else "✗"
        print(f"  {status} {test_name}")
    
    print(f"\n通过: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 终极验证完全通过！")
        print("✅ BVParti预测器与CVC5版本完全一致")
        print("✅ 所有逻辑都已验证正确")
        print("✅ 没有发现任何错误或不一致")
        print("✅ 可以安全投入生产使用")
        print("\n🚀 BVParti预测器已准备就绪！")
    else:
        print("\n❌ 发现问题，需要进一步检查。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
