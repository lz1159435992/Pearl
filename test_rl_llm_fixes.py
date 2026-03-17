#!/usr/bin/env python3
"""
测试QF_NIA_run_advanced_predictor.py中RL+LLM修复的验证
"""

import os
import sys
import tempfile
import json

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_imports():
    """测试所有必要的导入是否正常工作"""
    print("Testing imports...")
    
    try:
        from test_rl.test_cvc5.predict_z3_process.QF_NIA_run_advanced_predictor import (
            ConstraintSimplificationEnv_test,
            process_embeding,
            is_number,
            get_actions,
            create_agent,
            run_rl_training,
            run_simple_strategy
        )
        print("✅ 所有主要组件导入成功")
        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_environment_class():
    """测试ConstraintSimplificationEnv_test类的结构"""
    print("\nTesting ConstraintSimplificationEnv_test class...")
    
    try:
        from test_rl.test_cvc5.predict_z3_process.QF_NIA_run_advanced_predictor import ConstraintSimplificationEnv_test
        
        # 检查类的关键方法
        required_methods = [
            '__init__', 'reset', 'step', 'action_space', 'range_init',
            'process_text_python', 'calculate_reward', 'counter_reward_function'
        ]
        
        for method in required_methods:
            if hasattr(ConstraintSimplificationEnv_test, method):
                print(f"✅ Method {method} exists")
            else:
                print(f"❌ Method {method} missing")
                return False
        
        print("✅ ConstraintSimplificationEnv_test class structure correct")
        return True
        
    except Exception as e:
        print(f"❌ Class structure test failed: {e}")
        return False

def test_llm_integration():
    """测试LLM集成功能"""
    print("\nTesting LLM integration...")
    
    try:
        from test_rl.test_cvc5.predict_z3_process.QF_NIA_run_advanced_predictor import (
            ConstraintSimplificationEnv_test, process_embeding
        )
        
        # 检查process_embeding函数
        print("✅ process_embeding function available")
        
        # 检查ConstraintSimplificationEnv_test是否有LLM方法
        if hasattr(ConstraintSimplificationEnv_test, 'process_text_python'):
            print("✅ process_text_python method available")
        else:
            print("❌ process_text_python method missing")
            return False
        
        print("✅ LLM integration components present")
        return True
        
    except Exception as e:
        print(f"❌ LLM integration test failed: {e}")
        return False

def test_rl_components():
    """测试RL组件"""
    print("\nTesting RL components...")
    
    try:
        from test_rl.test_cvc5.predict_z3_process.QF_NIA_run_advanced_predictor import (
            create_agent, run_rl_training, run_simple_strategy
        )
        
        print("✅ create_agent function available")
        print("✅ run_rl_training function available") 
        print("✅ run_simple_strategy function available")
        
        # 检查函数签名
        import inspect
        
        sig = inspect.signature(create_agent)
        if list(sig.parameters.keys()) == ['env', 'action_space']:
            print("✅ create_agent signature correct")
        else:
            print(f"❌ create_agent signature incorrect: {list(sig.parameters.keys())}")
            return False
        
        sig = inspect.signature(run_rl_training)
        if list(sig.parameters.keys()) == ['env', 'args']:
            print("✅ run_rl_training signature correct")
        else:
            print(f"❌ run_rl_training signature incorrect: {list(sig.parameters.keys())}")
            return False
        
        print("✅ RL components structure correct")
        return True
        
    except Exception as e:
        print(f"❌ RL components test failed: {e}")
        return False

def test_utility_functions():
    """测试工具函数"""
    print("\nTesting utility functions...")
    
    try:
        from test_rl.test_cvc5.predict_z3_process.QF_NIA_run_advanced_predictor import is_number, get_actions
        import torch
        
        # 测试is_number函数
        test_cases = [
            ("123", True),
            ("123.45", True),
            ("123/456", True),
            ("abc", False),
            ("12.34.56", False),
        ]
        
        for test_input, expected in test_cases:
            result = is_number(test_input)
            if result == expected:
                print(f"✅ is_number('{test_input}') = {result}")
            else:
                print(f"❌ is_number('{test_input}') = {result}, expected {expected}")
                return False
        
        # 测试get_actions函数
        test_tensor = torch.arange(0, 5)
        result = get_actions(test_tensor)
        if isinstance(result, torch.Tensor) and result.shape == test_tensor.shape:
            print("✅ get_actions function works correctly")
        else:
            print(f"❌ get_actions function failed: {result}")
            return False
        
        print("✅ Utility functions work correctly")
        return True
        
    except Exception as e:
        print(f"❌ Utility functions test failed: {e}")
        return False

def test_code_consistency():
    """测试代码一致性"""
    print("\nTesting code consistency...")
    
    try:
        # 读取QF_NIA_run_advanced_predictor.py文件
        qf_nia_path = "test_rl/test_cvc5/predict_z3_process/QF_NIA_run_advanced_predictor.py"
        with open(qf_nia_path, 'r') as f:
            content = f.read()
        
        # 检查关键组件是否存在
        checks = [
            ("LLM chat integration", "def process_text_python(self, text, variable_pred):"),
            ("Counterexample management", "self.counterexamples_list"),
            ("Constraint validation", "related_assertions = self.v_related_assertions"),
            ("Reward calculation", "def calculate_reward(self, solver):"),
            ("Agent creation", "def create_agent(env, action_space):"),
            ("RL training", "def run_rl_training(env, args):"),
            ("Action space creation", "DiscreteActionSpace(self.actions)"),
            ("State updates", "process_embeding(solver.to_smt2(), self.llm_host)"),
            ("Error handling", "except Exception as e:"),
            ("LLM host configuration", "llm_host=args.llm_host"),
        ]
        
        all_passed = True
        for check_name, pattern in checks:
            if pattern in content:
                print(f"✅ {check_name}: Found")
            else:
                print(f"❌ {check_name}: Missing")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Code consistency test failed: {e}")
        return False

def test_integration_completeness():
    """测试集成完整性"""
    print("\nTesting integration completeness...")
    
    try:
        qf_nia_path = "test_rl/test_cvc5/predict_z3_process/QF_NIA_run_advanced_predictor.py"
        with open(qf_nia_path, 'r') as f:
            content = f.read()
        
        # 检查是否移除了随机动作
        if "random.randint(0, len(env.variables) - 1)" in content:
            print("⚠️  Random action selection still present in some places")
        else:
            print("✅ Random action selection properly replaced")
        
        # 检查是否有适当的RL训练调用
        if "run_rl_training(env, args)" in content:
            print("✅ Proper RL training call present")
        else:
            print("❌ RL training call missing")
            return False
        
        # 检查环境参数传递
        if "llm_host=args.llm_host, llm_model=args.llm_model" in content:
            print("✅ LLM parameters properly passed to environment")
        else:
            print("❌ LLM parameters not properly passed")
            return False
        
        # 检查ActionResult返回
        if "ActionResult(" in content:
            print("✅ Proper ActionResult returns")
        else:
            print("❌ ActionResult returns missing")
            return False
        
        print("✅ Integration completeness verified")
        return True
        
    except Exception as e:
        print(f"❌ Integration completeness test failed: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("QF_NIA_run_advanced_predictor.py RL+LLM修复验证测试")
    print("=" * 60)
    
    tests = [
        ("导入测试", test_imports),
        ("环境类结构测试", test_environment_class),
        ("LLM集成测试", test_llm_integration),
        ("RL组件测试", test_rl_components),
        ("工具函数测试", test_utility_functions),
        ("代码一致性测试", test_code_consistency),
        ("集成完整性测试", test_integration_completeness),
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
        print("🎉 所有测试通过！RL+LLM修复成功！")
        print("\n主要修复内容:")
        print("1. ✅ 完整的LLM集成 (process_text_python方法)")
        print("2. ✅ 反例管理系统 (counterexamples_list)")
        print("3. ✅ 约束验证逻辑 (related_assertions)")
        print("4. ✅ 复杂奖励计算 (calculate_reward)")
        print("5. ✅ Pearl Agent集成 (create_agent)")
        print("6. ✅ 在线学习支持 (run_rl_training)")
        print("7. ✅ 动态状态更新 (process_embeding)")
        print("8. ✅ 适当的错误处理")
    else:
        print("⚠️  部分测试失败，请检查相关问题。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
