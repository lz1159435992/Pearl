#!/usr/bin/env python3
"""
Test script for RQ5 Hybrid Screening Implementation

This script tests the basic functionality of the hybrid screening strategy.
"""

import os
import sys
import json
import tempfile

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def create_test_constraint():
    """创建一个测试约束"""
    test_constraint = {
        "smt_script": """
(set-logic QF_BV)
(declare-fun x () (_ BitVec 32))
(declare-fun y () (_ BitVec 32))
(declare-fun z () (_ BitVec 32))
(assert (= x (_ bv10 32)))
(assert (= y (_ bv20 32)))
(assert (= z (_ bv30 32)))
(assert (= (bvadd x y) z))
(check-sat)
"""
    }
    return test_constraint

def test_feature_extraction():
    """测试特征提取功能"""
    print("Testing feature extraction...")
    
    try:
        from rq5_hybrid_screening import HybridScreeningStrategy
        
        # 创建模拟参数
        args = type('Args', (), {
            'binary_model_path': '/home/lz/PycharmProjects/Pearl/test_rl/test_cvc5/predict_z3_process/models/bert_predictor_mask_best.pth',
            'eight_class_model_path': '/home/lz/PycharmProjects/Pearl/test_rl/test_cvc5/predict_z3_process/models/bert_predictor_2_mask_best_model.pth',
            'solver': 'z3',
            'timeout': 1200,
            'llm_host': 'http://172.29.7.221:32943',
            'llm_model': 'llama3.1:70b',
            'num_episodes': 1,
            'record_period': 1
        })()
        
        # 创建实验器（不加载模型，只测试特征提取）
        experiment = HybridScreeningStrategy(args)
        
        # 测试约束
        test_constraint = create_test_constraint()
        smtlib_str = test_constraint["smt_script"]
        
        # 提取特征
        features = experiment.extract_constraint_features(smtlib_str)
        
        print("Extracted features:")
        for key, value in features.items():
            print(f"  {key}: {value}")
        
        # 验证特征
        assert features['variable_count'] == 3, f"Expected 3 variables, got {features['variable_count']}"
        assert features['constraint_count'] == 4, f"Expected 4 constraints, got {features['constraint_count']}"
        
        print("✅ Feature extraction test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Feature extraction test failed: {e}")
        return False

def test_hybrid_screening_logic():
    """测试混合筛选逻辑（不依赖模型）"""
    print("Testing hybrid screening logic...")
    
    try:
        from rq5_hybrid_screening import HybridScreeningStrategy
        
        # 创建模拟参数
        args = type('Args', (), {
            'binary_model_path': '/home/lz/PycharmProjects/Pearl/test_rl/test_cvc5/predict_z3_process/models/bert_predictor_mask_best.pth',
            'eight_class_model_path': '/home/lz/PycharmProjects/Pearl/test_rl/test_cvc5/predict_z3_process/models/bert_predictor_2_mask_best_model.pth',
            'solver': 'z3',
            'timeout': 1200,
            'llm_host': 'http://172.29.7.221:32943',
            'llm_model': 'llama3.1:70b',
            'num_episodes': 1,
            'record_period': 1
        })()
        
        # 创建实验器
        experiment = HybridScreeningStrategy(args)
        
        # 测试简单约束
        simple_constraint = {
            "smt_script": """
(set-logic QF_LIA)
(declare-fun x () Int)
(assert (= x 10))
(check-sat)
"""
        }
        
        # 测试复杂约束
        complex_constraint = {
            "smt_script": """
(set-logic QF_BV)
(declare-fun x () (_ BitVec 64))
(declare-fun y () (_ BitVec 64))
(declare-fun z () (_ BitVec 64))
(declare-fun w () (_ BitVec 64))
(declare-fun v () (_ BitVec 64))
(assert (= (bvadd (bvmul x y) (bvsub z w)) v))
(assert (= (bvxor x y) (bvor z w)))
(assert (= (bvand x (bvnot y)) (bvshl z 2)))
(assert (= (bvlshr w 3) (bvashr v 1)))
(check-sat)
"""
        }
        
        # 测试特征筛选
        simple_features = experiment.extract_constraint_features(simple_constraint["smt_script"])
        complex_features = experiment.extract_constraint_features(complex_constraint["smt_script"])
        
        print(f"Simple constraint complexity: {simple_features['complexity_score']}")
        print(f"Complex constraint complexity: {complex_features['complexity_score']}")
        
        # 验证复杂约束的复杂度更高
        assert complex_features['complexity_score'] > simple_features['complexity_score'], \
            "Complex constraint should have higher complexity score"
        
        print("✅ Hybrid screening logic test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Hybrid screening logic test failed: {e}")
        return False

def test_file_structure():
    """测试文件结构"""
    print("Testing file structure...")
    
    required_files = [
        'rq5_hybrid_screening.py',
        'rq5_example_usage.py',
        'README_RQ5_Hybrid_Screening.md'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files exist!")
        return True

def test_imports():
    """测试导入功能"""
    print("Testing imports...")
    
    try:
        # 测试主要模块导入
        import rq5_hybrid_screening
        print("✅ Main module import successful")
        
        # 测试类导入
        from rq5_hybrid_screening import HybridScreeningStrategy
        print("✅ HybridScreeningStrategy class import successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False

def run_all_tests():
    """运行所有测试"""
    print("=" * 80)
    print("RQ5 Hybrid Screening Implementation Tests")
    print("=" * 80)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("Feature Extraction", test_feature_extraction),
        ("Hybrid Screening Logic", test_hybrid_screening_logic)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} test failed!")
    
    print("\n" + "=" * 80)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 80)
    
    if passed == total:
        print("🎉 All tests passed! RQ5 implementation is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 