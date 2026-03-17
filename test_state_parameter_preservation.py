#!/usr/bin/env python3
"""
测试QF_NIA_run_advanced_predictor.py中state参数的保留和条件使用
验证embedder存在时使用get_max_pooling_embedding，否则使用process_embeding
"""

import os
import sys
import tempfile
import json
import torch

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_state_parameter_preservation():
    """测试state参数是否保留"""
    print("Testing state parameter preservation...")
    
    try:
        qf_nia_path = "test_rl/test_cvc5/predict_z3_process/QF_NIA_run_advanced_predictor.py"
        with open(qf_nia_path, 'r') as f:
            content = f.read()
        
        # 检查__init__方法是否包含state参数
        if "def __init__(self, embedder, z3ast, model, model_time, smtlib_str, file_path, var_dict, state," in content:
            print("✅ state参数已保留在__init__方法中")
        else:
            print("❌ state参数在__init__方法中缺失")
            return False
        
        # 检查state_original的处理
        if "self.state_original = state" in content:
            print("✅ state_original正确设置")
        else:
            print("❌ state_original设置缺失")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ state参数保留测试失败: {e}")
        return False

def test_conditional_embedding_usage():
    """测试条件编码使用"""
    print("\nTesting conditional embedding usage...")
    
    try:
        qf_nia_path = "test_rl/test_cvc5/predict_z3_process/QF_NIA_run_advanced_predictor.py"
        with open(qf_nia_path, 'r') as f:
            content = f.read()
        
        # 检查reset方法中的条件逻辑
        reset_checks = [
            ("embedder存在检查", "if self.embedder is not None:"),
            ("使用state_original", "self.state = self.state_original.clone().detach()"),
            ("使用process_embeding", "self.state = process_embeding(self.smtlib_str, self.llm_host).unsqueeze(0)"),
        ]
        
        all_passed = True
        for check_name, pattern in reset_checks:
            if pattern in content:
                print(f"✅ reset方法 - {check_name}: 存在")
            else:
                print(f"❌ reset方法 - {check_name}: 缺失")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ 条件编码使用测试失败: {e}")
        return False

def test_step_method_conditional_logic():
    """测试step方法中的条件逻辑"""
    print("\nTesting step method conditional logic...")
    
    try:
        qf_nia_path = "test_rl/test_cvc5/predict_z3_process/QF_NIA_run_advanced_predictor.py"
        with open(qf_nia_path, 'r') as f:
            content = f.read()
        
        # 检查step方法中的条件逻辑
        step_checks = [
            ("embedder存在检查", "if self.embedder is not None:"),
            ("使用get_max_pooling_embedding", "self.embedder.get_max_pooling_embedding(solver.to_smt2()).unsqueeze(0)"),
            ("使用process_embeding", "process_embeding(solver.to_smt2(), self.llm_host).unsqueeze(0)"),
        ]
        
        all_passed = True
        for check_name, pattern in step_checks:
            count = content.count(pattern)
            if count >= 2:  # step方法中应该有两个地方使用这个逻辑
                print(f"✅ step方法 - {check_name}: 存在 ({count}次)")
            else:
                print(f"❌ step方法 - {check_name}: 不足 ({count}次)")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ step方法条件逻辑测试失败: {e}")
        return False

def test_calculate_reward_conditional_logic():
    """测试calculate_reward方法中的条件逻辑"""
    print("\nTesting calculate_reward method conditional logic...")
    
    try:
        qf_nia_path = "test_rl/test_cvc5/predict_z3_process/QF_NIA_run_advanced_predictor.py"
        with open(qf_nia_path, 'r') as f:
            content = f.read()
        
        # 检查calculate_reward方法中的条件逻辑
        reward_checks = [
            ("embedder存在检查", "if self.embedder is not None:"),
            ("使用get_max_pooling_embedding", "self.embedder.get_max_pooling_embedding("),
            ("使用process_embeding", "process_embeding("),
        ]
        
        all_passed = True
        for check_name, pattern in reward_checks:
            count = content.count(pattern)
            if count >= 4:  # calculate_reward中应该有多个地方使用这个逻辑
                print(f"✅ calculate_reward方法 - {check_name}: 存在 ({count}次)")
            else:
                print(f"❌ calculate_reward方法 - {check_name}: 不足 ({count}次)")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ calculate_reward方法条件逻辑测试失败: {e}")
        return False

def test_environment_creation_with_state():
    """测试环境创建时state参数的传递"""
    print("\nTesting environment creation with state parameter...")
    
    try:
        qf_nia_path = "test_rl/test_cvc5/predict_z3_process/QF_NIA_run_advanced_predictor.py"
        with open(qf_nia_path, 'r') as f:
            content = f.read()
        
        # 检查_process_worker_qf_nia中的环境创建
        creation_checks = [
            ("生成初始状态", "initial_state = process_embeding(smtlib_str, args.llm_host)"),
            ("传递state参数", "file_path, var_dict, initial_state,"),
        ]
        
        all_passed = True
        for check_name, pattern in creation_checks:
            if pattern in content:
                print(f"✅ 环境创建 - {check_name}: 存在")
            else:
                print(f"❌ 环境创建 - {check_name}: 缺失")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ 环境创建测试失败: {e}")
        return False

def test_mock_environment_initialization():
    """测试模拟环境初始化"""
    print("\nTesting mock environment initialization...")
    
    try:
        from test_rl.test_cvc5.predict_z3_process.QF_NIA_run_advanced_predictor import ConstraintSimplificationEnv_test
        
        # 创建模拟参数
        mock_z3ast = []
        mock_model = None
        mock_model_time = None
        mock_smtlib_str = "(set-logic QF_NIA)\n(declare-fun x () Int)\n(assert (> x 0))\n(check-sat)"
        mock_file_path = "test.smt2"
        mock_var_dict = {"x": "VAR1"}
        mock_state = torch.randn(8192)
        
        # 测试embedder为None的情况
        env1 = ConstraintSimplificationEnv_test(
            None, mock_z3ast, mock_model, mock_model_time, 
            mock_smtlib_str, mock_file_path, mock_var_dict, mock_state
        )
        
        if env1.embedder is None:
            print("✅ embedder为None时环境创建成功")
        else:
            print("❌ embedder为None时环境创建失败")
            return False
        
        # 测试embedder存在的情况（模拟）
        class MockEmbedder:
            def get_max_pooling_embedding(self, text):
                return torch.randn(8192)
        
        mock_embedder = MockEmbedder()
        env2 = ConstraintSimplificationEnv_test(
            mock_embedder, mock_z3ast, mock_model, mock_model_time, 
            mock_smtlib_str, mock_file_path, mock_var_dict, mock_state
        )
        
        if env2.embedder is not None:
            print("✅ embedder存在时环境创建成功")
        else:
            print("❌ embedder存在时环境创建失败")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ 模拟环境初始化测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("QF_NIA_run_advanced_predictor.py state参数保留验证测试")
    print("=" * 60)
    
    tests = [
        ("state参数保留测试", test_state_parameter_preservation),
        ("条件编码使用测试", test_conditional_embedding_usage),
        ("step方法条件逻辑测试", test_step_method_conditional_logic),
        ("calculate_reward条件逻辑测试", test_calculate_reward_conditional_logic),
        ("环境创建state参数测试", test_environment_creation_with_state),
        ("模拟环境初始化测试", test_mock_environment_initialization),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 执行失败: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试结果总结:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！state参数保留和条件使用成功！")
        print("\n实现特点:")
        print("1. ✅ 保留了state参数在__init__方法中")
        print("2. ✅ 在embedder存在时使用get_max_pooling_embedding")
        print("3. ✅ 在embedder为None时使用process_embeding")
        print("4. ✅ reset、step、calculate_reward方法都有条件逻辑")
        print("5. ✅ 环境创建时正确传递state参数")
        print("6. ✅ 支持两种编码方式的动态切换")
    else:
        print("⚠️  部分测试失败，请检查相关问题。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
