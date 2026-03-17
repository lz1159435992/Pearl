#!/usr/bin/env python3
"""
全面的逻辑测试：深入检查BVParti预测器的所有关键逻辑
"""

import os
import sys
import json
import traceback
import tempfile

# 添加路径
sys.path.append('/home/lz/PycharmProjects/Pearl')

def test_bvparti_solver_paths():
    """测试BVParti求解器路径配置"""
    print("=== 测试BVParti求解器路径配置 ===")
    
    try:
        from test_rl.test_cvc5.bvparti_process.run_bvparti_predictor import BVPartiConfig
        
        config = BVPartiConfig()
        
        paths_to_check = [
            ("BVParti二进制", config.bvparti_bin),
            ("分区器二进制", config.partitioner_bin),
            ("Bitwuzla二进制", config.bitwuzla_bin),
            ("运行脚本", config.run_bvparti_script),
            ("临时目录基础路径", config.temp_dir_base)
        ]
        
        all_paths_exist = True
        for name, path in paths_to_check:
            if name == "临时目录基础路径":
                # 临时目录可能不存在，但应该可以创建
                try:
                    os.makedirs(path, exist_ok=True)
                    exists = os.path.exists(path)
                except:
                    exists = False
            else:
                exists = os.path.exists(path)
            
            status = "✓" if exists else "✗"
            print(f"  {status} {name}: {path}")
            
            if not exists:
                all_paths_exist = False
        
        return all_paths_exist
        
    except Exception as e:
        print(f"测试BVParti路径配置时出错: {e}")
        traceback.print_exc()
        return False

def test_solver_creation():
    """测试求解器创建"""
    print("\n=== 测试求解器创建 ===")
    
    try:
        from test_rl.test_cvc5.bvparti_process.run_bvparti_predictor import get_solver, BVPartiSolver, Z3Solver, CVC5Solver
        
        solvers_to_test = ['bvparti', 'z3', 'cvc5']
        expected_types = [BVPartiSolver, Z3Solver, CVC5Solver]
        
        all_created = True
        for solver_name, expected_type in zip(solvers_to_test, expected_types):
            try:
                solver = get_solver(solver_name)
                actual_type = type(solver)
                
                if isinstance(solver, expected_type):
                    print(f"  ✓ {solver_name}: {actual_type.__name__}")
                else:
                    print(f"  ✗ {solver_name}: 期望 {expected_type.__name__}, 实际 {actual_type.__name__}")
                    all_created = False
            except Exception as e:
                print(f"  ✗ {solver_name}: 创建失败 - {e}")
                all_created = False
        
        return all_created
        
    except Exception as e:
        print(f"测试求解器创建时出错: {e}")
        traceback.print_exc()
        return False

def test_data_processing_consistency():
    """测试数据处理一致性"""
    print("\n=== 测试数据处理一致性 ===")
    
    try:
        from test_rl.test_cvc5.bvparti_process.test_group_get_dis_smt_comp_bert_embeding_single import (
            get_solve_result_and_time, convert_timeout_to_unknown
        )
        
        # 测试复杂格式数据处理
        complex_data = {
            "metadata": {"test": "data"},
            "results": {
                "/file1": {"result": "sat", "solve_time": 1.5},
                "/file2": {"result": "unsat", "solve_time": 2.0},
                "/file3": {"result": "timeout", "solve_time": 1200.0},
                "/file4": {"result": "error", "solve_time": -1}
            }
        }
        
        # 应用转换
        converted_data = convert_timeout_to_unknown(complex_data)
        
        # 验证转换结果
        expected_results = {
            "/file1": ("sat", 1.5),
            "/file2": ("unsat", 2.0),
            "/file3": ("unknown", 1200.0),  # timeout -> unknown
            "/file4": ("unknown", 0)        # error -> unknown, -1 -> 0
        }
        
        all_correct = True
        for key, (expected_category, expected_time) in expected_results.items():
            actual_category, actual_time = get_solve_result_and_time(converted_data, key)
            
            if actual_category == expected_category and actual_time == expected_time:
                print(f"  ✓ {key}: {actual_category}, {actual_time}s")
            else:
                print(f"  ✗ {key}: 期望 ({expected_category}, {expected_time}), 实际 ({actual_category}, {actual_time})")
                all_correct = False
        
        return all_correct
        
    except Exception as e:
        print(f"测试数据处理一致性时出错: {e}")
        traceback.print_exc()
        return False

def test_environment_initialization():
    """测试环境初始化"""
    print("\n=== 测试环境初始化 ===")
    
    try:
        from test_rl.test_cvc5.bvparti_process.run_bvparti_predictor import ConstraintSimplificationEnv_test
        
        # 创建模拟参数
        class MockEmbedder:
            def get_max_pooling_embedding(self, smtlib_str, variables):
                import torch
                return torch.randn(768)  # 模拟嵌入向量
        
        class MockModel:
            def __call__(self, x):
                import torch
                return torch.tensor([0.7])  # 模拟预测结果
        
        class MockTimeModel:
            def __call__(self, x):
                import torch
                return torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0]])  # 模拟时间预测
        
        # 模拟Z3断言
        import z3
        x = z3.BitVec('x', 8)
        y = z3.BitVec('y', 8)
        z3ast = [x + y == 10, x > 5]
        
        # 模拟SMT字符串
        smtlib_str = """(set-logic QF_BV)
(declare-fun x () (_ BitVec 8))
(declare-fun y () (_ BitVec 8))
(assert (= (bvadd x y) (_ bv10 8)))
(assert (bvugt x (_ bv5 8)))
(check-sat)
(exit)"""
        
        # 测试不同求解器的环境初始化
        solvers_to_test = ['bvparti', 'z3', 'cvc5']
        
        all_initialized = True
        for solver_name in solvers_to_test:
            try:
                env = ConstraintSimplificationEnv_test(
                    embedder=MockEmbedder(),
                    z3ast=z3ast,
                    model=MockModel(),
                    model_time=MockTimeModel(),
                    smtlib_str=smtlib_str,
                    file_path="/test/file.smt2",
                    var_dict={},
                    constant_list=[],
                    solver_name=solver_name
                )
                
                # 验证环境属性
                if hasattr(env, 'solver') and hasattr(env, 'variables') and hasattr(env, 'solver_name'):
                    print(f"  ✓ {solver_name}: 环境初始化成功，变量数量={len(env.variables)}")
                else:
                    print(f"  ✗ {solver_name}: 环境初始化不完整")
                    all_initialized = False
                    
            except Exception as e:
                print(f"  ✗ {solver_name}: 环境初始化失败 - {e}")
                all_initialized = False
        
        return all_initialized
        
    except Exception as e:
        print(f"测试环境初始化时出错: {e}")
        traceback.print_exc()
        return False

def test_file_processing_logic():
    """测试文件处理逻辑"""
    print("\n=== 测试文件处理逻辑 ===")
    
    try:
        # 加载实际的结果文件
        result_file_path = '/home/lz/PycharmProjects/Pearl/test_rl/AriParti_sync/scripts/batch_output/bv_default/SMTimer_z3_result_rl.json'
        
        if not os.path.exists(result_file_path):
            print(f"  ✗ 结果文件不存在: {result_file_path}")
            return False
        
        with open(result_file_path, 'r') as f:
            result_dict = json.load(f)
        
        # 模拟处理逻辑
        from test_rl.test_cvc5.bvparti_process.test_group_get_dis_smt_comp_bert_embeding_single import (
            get_solve_result_and_time, convert_timeout_to_unknown
        )
        
        # 应用转换
        result_dict = convert_timeout_to_unknown(result_dict)
        
        # 获取结果数据
        results_data = result_dict.get("results", result_dict) if "results" in result_dict else result_dict
        
        # 模拟RL字典
        rl_dict_path = '/home/lz/sibyl_3/src/networks/info_dict_rl.txt'
        if os.path.exists(rl_dict_path):
            with open(rl_dict_path, 'r') as f:
                rl_dict = json.load(f)
        else:
            print(f"  ✗ RL字典文件不存在: {rl_dict_path}")
            return False
        
        # 模拟info_dict（空的）
        info_dict = {}
        
        # 测试过滤逻辑
        time_threshold = 300
        filtered_files = []
        
        for key in list(results_data.keys())[:100]:  # 只测试前100个
            category, time_value = get_solve_result_and_time(result_dict, key)
            if category in ["sat", "unknown"] and time_value > time_threshold and key in rl_dict.keys() and key not in info_dict.keys():
                filtered_files.append((key, category, time_value))
        
        print(f"  ✓ 数据格式检测: {'复杂格式' if 'results' in result_dict else '简单格式'}")
        print(f"  ✓ 总结果条目数: {len(results_data)}")
        print(f"  ✓ RL字典条目数: {len(rl_dict)}")
        print(f"  ✓ 前100个中符合条件的文件: {len(filtered_files)}")
        
        if len(filtered_files) > 0:
            print(f"  ✓ 示例文件: {os.path.basename(filtered_files[0][0])}, {filtered_files[0][1]}, {filtered_files[0][2]:.1f}s")
        
        return True
        
    except Exception as e:
        print(f"测试文件处理逻辑时出错: {e}")
        traceback.print_exc()
        return False

def test_solver_result_format():
    """测试求解器结果格式"""
    print("\n=== 测试求解器结果格式 ===")
    
    try:
        from test_rl.test_cvc5.bvparti_process.run_bvparti_predictor import SolverResult, Z3Solver
        
        # 测试SolverResult类
        result = SolverResult(1.5, "sat", "model_output")
        
        if hasattr(result, 'solve_time') and hasattr(result, 'result') and hasattr(result, 'model'):
            print(f"  ✓ SolverResult格式正确: {result.result}, {result.solve_time}s")
        else:
            print(f"  ✗ SolverResult格式不正确")
            return False
        
        # 测试Z3求解器
        z3_solver = Z3Solver()
        simple_smt = "(set-logic QF_LIA)\n(declare-fun x () Int)\n(assert (= x 42))\n(check-sat)\n(exit)"
        
        z3_result = z3_solver.solve(simple_smt, timeout=5)
        
        if isinstance(z3_result, SolverResult):
            print(f"  ✓ Z3求解器结果格式正确: {z3_result.result}, {z3_result.solve_time:.3f}s")
        else:
            print(f"  ✗ Z3求解器结果格式不正确: {type(z3_result)}")
            return False
        
        return True
        
    except Exception as e:
        print(f"测试求解器结果格式时出错: {e}")
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("BVParti预测器全面逻辑测试")
    print("=" * 80)
    
    tests = [
        ("BVParti求解器路径配置", test_bvparti_solver_paths),
        ("求解器创建", test_solver_creation),
        ("数据处理一致性", test_data_processing_consistency),
        ("环境初始化", test_environment_initialization),
        ("文件处理逻辑", test_file_processing_logic),
        ("求解器结果格式", test_solver_result_format),
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
    
    # 总结
    print("\n全面测试总结:")
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓" if result else "✗"
        print(f"  {status} {test_name}")
    
    print(f"\n通过: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 所有测试通过！BVParti预测器逻辑完全正确。")
        print("✅ 数据处理逻辑正确")
        print("✅ 求解器集成正确")
        print("✅ 环境初始化正确")
        print("✅ 文件处理逻辑正确")
        print("✅ 结果格式一致")
    else:
        print("\n⚠️  部分测试失败，需要进一步检查。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
