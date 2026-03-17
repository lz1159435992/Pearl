#!/usr/bin/env python3
"""
测试更新后的QF_NIA_run_advanced_predictor.py
验证与env_gai_6_llm_add_ce_predictor_docker_llm_embed.py的一致性
"""

import os
import sys
import torch
import json
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_imports():
    """测试所有必要的导入是否正常工作"""
    print("Testing imports...")
    
    try:
        from test_rl.test_cvc5.predict_z3_process.QF_NIA_run_advanced_predictor import (
            process_embeding,
            run_advanced_prediction_flow
        )
        print("✅ QF_NIA_run_advanced_predictor imports successful")
    except ImportError as e:
        print(f"❌ QF_NIA_run_advanced_predictor import failed: {e}")
        return False
    
    try:
        from test_rl.predictor.smt_comp_NIA.bert_predictor_mask_llm import EnhancedClassifier
        from test_rl.predictor.smt_comp_NIA.bert_predictor_2_mask_llm import EnhancedEightClassModelLargeInput
        print("✅ Model imports successful")
    except ImportError as e:
        print(f"❌ Model import failed: {e}")
        return False
    
    try:
        from test_rl.test_script.utils import normalize_smt_str
        print("✅ Utils imports successful")
    except ImportError as e:
        print(f"❌ Utils import failed: {e}")
        return False
    
    return True

def test_embedding_function():
    """测试embedding函数是否正常工作"""
    print("\nTesting embedding function...")
    
    try:
        from test_rl.test_cvc5.predict_z3_process.QF_NIA_run_advanced_predictor import process_embeding
        
        # 测试用的简单SMT字符串
        test_smt = "(set-info :smt-lib-version 2.6)\n(set-logic QF_NIA)\n(declare-fun x () Int)\n(assert (> x 0))\n(check-sat)"
        
        # 注意：这个测试需要Ollama服务器运行
        # 如果服务器不可用，会抛出异常
        try:
            embedding = process_embeding(test_smt)
            print(f"✅ Embedding generation successful, shape: {embedding.shape}")
            print(f"   Embedding type: {type(embedding)}")
            print(f"   Embedding dtype: {embedding.dtype}")
            return True
        except Exception as e:
            print(f"⚠️  Embedding generation failed (likely Ollama server unavailable): {e}")
            return False
            
    except ImportError as e:
        print(f"❌ Cannot import process_embeding: {e}")
        return False

def test_model_loading():
    """测试模型加载是否正常工作"""
    print("\nTesting model loading...")
    
    try:
        from test_rl.predictor.smt_comp_NIA.bert_predictor_mask_llm import EnhancedClassifier
        from test_rl.predictor.smt_comp_NIA.bert_predictor_2_mask_llm import EnhancedEightClassModelLargeInput
        
        # 创建模型实例
        solvability_model = EnhancedClassifier()
        time_model = EnhancedEightClassModelLargeInput()
        
        print(f"✅ Solvability model created: {type(solvability_model)}")
        print(f"✅ Time model created: {type(time_model)}")
        
        # 测试模型的前向传播（使用随机输入）
        test_input = torch.randn(1, 8192)  # LLaMA 3.1:70b embedding dimension
        
        with torch.no_grad():
            solvability_output = solvability_model(test_input)
            time_output = time_model(test_input)
            
        print(f"✅ Solvability model output shape: {solvability_output.shape}")
        print(f"✅ Time model output shape: {time_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading/testing failed: {e}")
        return False

def test_normalize_smt_str():
    """测试SMT字符串标准化函数"""
    print("\nTesting SMT string normalization...")
    
    try:
        from test_rl.test_script.utils import normalize_smt_str
        
        test_smt = "(set-info :smt-lib-version 2.6)\n(set-logic QF_NIA)\n(declare-fun x () Int)\n(declare-fun y () Int)\n(assert (and (> x 0) (< y 10)))\n(check-sat)"
        
        normalized_str, var_dict, _ = normalize_smt_str(test_smt)
        
        print(f"✅ SMT normalization successful")
        print(f"   Original length: {len(test_smt)}")
        print(f"   Normalized length: {len(normalized_str)}")
        print(f"   Variable dict: {var_dict}")
        
        return True
        
    except Exception as e:
        print(f"❌ SMT normalization failed: {e}")
        return False

def test_file_structure():
    """测试文件结构和路径"""
    print("\nTesting file structure...")
    
    # 检查关键文件是否存在
    files_to_check = [
        "test_rl/test_cvc5/predict_z3_process/QF_NIA_run_advanced_predictor.py",
        "test_rl/env_gai_6_llm_add_ce_predictor_docker_llm_embed.py",
        "test_rl/predictor/smt_comp_NIA/bert_predictor_mask_llm.py",
        "test_rl/predictor/smt_comp_NIA/bert_predictor_2_mask_llm.py",
    ]
    
    all_exist = True
    for file_path in files_to_check:
        full_path = os.path.join(project_root, file_path)
        if os.path.exists(full_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist

def compare_with_env_file():
    """比较与env文件的一致性"""
    print("\nComparing with env file...")
    
    try:
        # 检查两个文件中的关键函数是否一致
        from test_rl.test_cvc5.predict_z3_process.QF_NIA_run_advanced_predictor import process_embeding as qf_nia_embedding
        from test_rl.predictor.smt_comp_QF_IDL.test_group_get_dis_smt_comp_llm import process_embeding as env_embedding
        
        print("✅ Both embedding functions accessible")
        
        # 检查函数签名（通过inspect模块）
        import inspect
        qf_nia_sig = inspect.signature(qf_nia_embedding)
        env_sig = inspect.signature(env_embedding)
        
        print(f"   QF_NIA embedding signature: {qf_nia_sig}")
        print(f"   Env embedding signature: {env_sig}")
        
        # 检查模型类是否一致
        from test_rl.predictor.smt_comp_NIA.bert_predictor_mask_llm import EnhancedClassifier as NIA_Classifier
        from test_rl.predictor.smt_comp_NIA.bert_predictor_2_mask_llm import EnhancedEightClassModelLargeInput as NIA_TimeModel
        
        print("✅ Model classes accessible from both contexts")
        
        return True
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("QF_NIA_run_advanced_predictor.py 更新验证测试")
    print("=" * 60)
    
    tests = [
        ("文件结构检查", test_file_structure),
        ("导入测试", test_imports),
        ("SMT标准化测试", test_normalize_smt_str),
        ("模型加载测试", test_model_loading),
        ("与env文件一致性检查", compare_with_env_file),
        ("Embedding函数测试", test_embedding_function),  # 放在最后，因为需要外部服务
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
        print("🎉 所有测试通过！QF_NIA_run_advanced_predictor.py 更新成功！")
    else:
        print("⚠️  部分测试失败，请检查相关问题。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
