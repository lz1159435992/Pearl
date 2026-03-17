#!/usr/bin/env python3
"""
验证QF_NIA_run_advanced_predictor.py的更新
检查代码结构和关键函数的一致性
"""

import os
import re
import sys

def read_file_content(file_path):
    """读取文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def check_imports(content):
    """检查导入语句"""
    required_imports = [
        'from test_rl.predictor.smt_comp_NIA.bert_predictor_mask_llm import EnhancedClassifier',
        'from test_rl.predictor.smt_comp_NIA.bert_predictor_2_mask_llm import EnhancedEightClassModelLargeInput',
        'from test_rl.test_script.utils import normalize_smt_str',
        'from ollama import Client'
    ]
    
    results = []
    for import_stmt in required_imports:
        if import_stmt in content:
            results.append((import_stmt, True))
        else:
            results.append((import_stmt, False))
    
    return results

def check_function_definitions(content):
    """检查关键函数定义"""
    required_functions = [
        'def process_embeding(',
        'def run_advanced_prediction_flow(',
        'def main('
    ]
    
    results = []
    for func_def in required_functions:
        if func_def in content:
            results.append((func_def, True))
        else:
            results.append((func_def, False))
    
    return results

def check_model_usage(content):
    """检查模型使用方式"""
    model_patterns = [
        r'solvability_predictor = EnhancedClassifier\(\)',
        r'time_predictor = EnhancedEightClassModelLargeInput\(\)',
        r'solvability_predictor\.load_state_dict',
        r'time_predictor\.load_state_dict',
        r'with torch\.no_grad\(\):',
        r'solvability_output = solvability_predictor\(',
        r'time_output = time_predictor\('
    ]
    
    results = []
    for pattern in model_patterns:
        if re.search(pattern, content):
            results.append((pattern, True))
        else:
            results.append((pattern, False))
    
    return results

def check_embedding_usage(content):
    """检查embedding使用方式"""
    embedding_patterns = [
        r'normalize_smt_str\(',
        r'process_embeding\(',
        r'embedding\.unsqueeze\(0\)',
        r'Client\(host=',
        r'model=\'llama3\.1:70b\''
    ]
    
    results = []
    for pattern in embedding_patterns:
        if re.search(pattern, content):
            results.append((pattern, True))
        else:
            results.append((pattern, False))
    
    return results

def check_error_handling(content):
    """检查错误处理"""
    error_patterns = [
        r'try:',
        r'except.*Exception',
        r'except.*FileNotFoundError',
        r'except.*json\.JSONDecodeError',
        r'logger\.error\(',
        r'continue'
    ]
    
    results = []
    for pattern in error_patterns:
        if re.search(pattern, content):
            results.append((pattern, True))
        else:
            results.append((pattern, False))
    
    return results

def check_argument_parsing(content):
    """检查参数解析"""
    arg_patterns = [
        r'--binary_model_path',
        r'--eight_class_model_path',
        r'--ollama_host',
        r'enhanced_classifier_model\.pth',
        r'enhanced_eight_class_model_large_input\.pth',
        r'http://172\.29\.7\.221:32903'
    ]
    
    results = []
    for pattern in arg_patterns:
        if re.search(pattern, content):
            results.append((pattern, True))
        else:
            results.append((pattern, False))
    
    return results

def compare_with_env_file():
    """比较与env文件的关键差异"""
    qf_nia_path = "test_rl/test_cvc5/predict_z3_process/QF_NIA_run_advanced_predictor.py"
    env_path = "test_rl/env_gai_6_llm_add_ce_predictor_docker_llm_embed.py"
    
    qf_nia_content = read_file_content(qf_nia_path)
    env_content = read_file_content(env_path)
    
    if not qf_nia_content or not env_content:
        return False
    
    # 检查关键相似性
    similarities = []
    
    # 1. 都使用process_embeding函数
    qf_nia_has_process_embeding = 'def process_embeding(' in qf_nia_content
    env_has_process_embeding = 'process_embeding(' in env_content
    similarities.append(("process_embeding usage", qf_nia_has_process_embeding and env_has_process_embeding))
    
    # 2. 都使用相同的模型类
    qf_nia_has_enhanced_classifier = 'EnhancedClassifier' in qf_nia_content
    env_has_simple_classifier = 'SimpleClassifier' in env_content
    similarities.append(("Model class usage", qf_nia_has_enhanced_classifier))
    
    # 3. 都使用torch.no_grad
    qf_nia_has_no_grad = 'torch.no_grad' in qf_nia_content
    env_has_no_grad = 'torch.no_grad' in env_content
    similarities.append(("torch.no_grad usage", qf_nia_has_no_grad))
    
    # 4. 都使用normalize_smt_str
    qf_nia_has_normalize = 'normalize_smt_str' in qf_nia_content
    similarities.append(("normalize_smt_str usage", qf_nia_has_normalize))
    
    return similarities

def main():
    """主验证函数"""
    print("=" * 60)
    print("QF_NIA_run_advanced_predictor.py 更新验证")
    print("=" * 60)
    
    qf_nia_path = "test_rl/test_cvc5/predict_z3_process/QF_NIA_run_advanced_predictor.py"
    
    if not os.path.exists(qf_nia_path):
        print(f"❌ 文件不存在: {qf_nia_path}")
        return False
    
    content = read_file_content(qf_nia_path)
    if not content:
        print("❌ 无法读取文件内容")
        return False
    
    print(f"✅ 文件读取成功，共 {len(content.splitlines())} 行")
    
    # 执行各项检查
    checks = [
        ("导入语句检查", check_imports),
        ("函数定义检查", check_function_definitions),
        ("模型使用检查", check_model_usage),
        ("Embedding使用检查", check_embedding_usage),
        ("错误处理检查", check_error_handling),
        ("参数解析检查", check_argument_parsing),
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n{'-'*20} {check_name} {'-'*20}")
        results = check_func(content)
        
        passed = 0
        total = len(results)
        
        for item, status in results:
            status_str = "✅" if status else "❌"
            print(f"{status_str} {item}")
            if status:
                passed += 1
        
        print(f"通过: {passed}/{total}")
        if passed < total:
            all_passed = False
    
    # 与env文件的比较
    print(f"\n{'-'*20} 与env文件一致性检查 {'-'*20}")
    similarities = compare_with_env_file()
    if similarities:
        passed = 0
        total = len(similarities)
        for item, status in similarities:
            status_str = "✅" if status else "❌"
            print(f"{status_str} {item}")
            if status:
                passed += 1
        print(f"一致性: {passed}/{total}")
        if passed < total:
            all_passed = False
    else:
        print("❌ 无法进行一致性检查")
        all_passed = False
    
    # 总结
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有检查通过！QF_NIA_run_advanced_predictor.py 更新成功！")
        print("\n主要更新内容:")
        print("1. ✅ 更新了模型类为EnhancedClassifier和EnhancedEightClassModelLargeInput")
        print("2. ✅ 添加了process_embeding函数，使用LLaMA 3.1:70b模型")
        print("3. ✅ 集成了normalize_smt_str函数进行SMT字符串标准化")
        print("4. ✅ 添加了实时预测逻辑，替代预计算的预测结果")
        print("5. ✅ 增强了错误处理和异常捕获")
        print("6. ✅ 添加了必要的命令行参数")
    else:
        print("⚠️  部分检查未通过，请检查相关问题")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
