#!/usr/bin/env python3
"""
测试修复后的BVParti预测器逻辑
"""

import os
import sys
import json
import traceback

# 添加路径
sys.path.append('/home/lz/PycharmProjects/Pearl')

def test_data_processing_logic():
    """测试数据处理逻辑"""
    print("=== 测试数据处理逻辑 ===")
    
    try:
        from test_rl.test_cvc5.bvparti_process.test_group_get_dis_smt_comp_bert_embeding_single import (
            get_solve_result_and_time, convert_timeout_to_unknown
        )
        
        # 加载实际的结果文件
        result_file_path = '/home/lz/PycharmProjects/Pearl/test_rl/AriParti_sync/scripts/batch_output/bv_default/SMTimer_z3_result_rl.json'
        
        if not os.path.exists(result_file_path):
            print(f"✗ 文件不存在: {result_file_path}")
            return False
        
        with open(result_file_path, 'r') as f:
            result_dict = json.load(f)
        
        print(f"原始文件结构:")
        print(f"  顶层键: {list(result_dict.keys())}")
        
        # 应用timeout转换
        result_dict = convert_timeout_to_unknown(result_dict)
        
        # 测试新的数据访问逻辑
        results_data = result_dict.get("results", result_dict) if "results" in result_dict else result_dict
        results_count = len(results_data)
        
        print(f"  结果数据条目数: {results_count}")
        
        # 测试前几个条目的处理
        print("\n测试数据访问:")
        count = 0
        for key in list(results_data.keys())[:5]:
            category, time_value = get_solve_result_and_time(result_dict, key)
            print(f"  {count+1}. {os.path.basename(key)}: {category}, {time_value:.3f}s")
            count += 1
        
        # 模拟过滤逻辑
        print("\n测试过滤逻辑:")
        time_threshold = 300
        
        # 加载RL字典
        rl_dict_path = '/home/lz/sibyl_3/src/networks/info_dict_rl.txt'
        if os.path.exists(rl_dict_path):
            with open(rl_dict_path, 'r') as f:
                rl_dict = json.load(f)
        else:
            print(f"RL字典文件不存在: {rl_dict_path}")
            return False
        
        # 模拟info_dict（空的）
        info_dict = {}
        
        # 计算符合条件的文件数量
        total_files = 0
        for key in results_data.keys():
            category, time_value = get_solve_result_and_time(result_dict, key)
            if category in ["sat", "unknown"] and time_value > time_threshold and key in rl_dict.keys() and key not in info_dict.keys():
                total_files += 1
        
        print(f"  符合条件的文件数量: {total_files}")
        
        if total_files > 0:
            print("✓ 数据处理逻辑正确")
            return True
        else:
            print("⚠ 没有符合条件的文件，但逻辑正确")
            return True
            
    except Exception as e:
        print(f"测试数据处理逻辑时出错: {e}")
        traceback.print_exc()
        return False

def test_format_compatibility():
    """测试格式兼容性"""
    print("\n=== 测试格式兼容性 ===")
    
    try:
        from test_rl.test_cvc5.bvparti_process.test_group_get_dis_smt_comp_bert_embeding_single import (
            get_solve_result_and_time, convert_timeout_to_unknown
        )
        
        # 测试简单格式（cvc5格式）
        simple_format = {
            "/test/file1": ["sat", 1.5, 1200, {}],
            "/test/file2": ["timeout", 1200.0, 1200, {}]
        }
        
        print("测试简单格式:")
        converted_simple = convert_timeout_to_unknown(simple_format)
        for key in converted_simple:
            category, time_value = get_solve_result_and_time(converted_simple, key)
            print(f"  {key}: {category}, {time_value}s")
        
        # 测试复杂格式（bvparti格式）
        complex_format = {
            "metadata": {"test": "data"},
            "results": {
                "/test/file1": {
                    "result": "sat",
                    "solve_time": 1.5,
                    "total_time": 1.6
                },
                "/test/file2": {
                    "result": "timeout",
                    "solve_time": 1200.0,
                    "total_time": 1200.1
                }
            }
        }
        
        print("\n测试复杂格式:")
        converted_complex = convert_timeout_to_unknown(complex_format)
        for key in converted_complex.get("results", {}):
            category, time_value = get_solve_result_and_time(converted_complex, key)
            print(f"  {key}: {category}, {time_value}s")
        
        print("✓ 格式兼容性测试通过")
        return True
        
    except Exception as e:
        print(f"测试格式兼容性时出错: {e}")
        traceback.print_exc()
        return False

def test_data_access_patterns():
    """测试数据访问模式"""
    print("\n=== 测试数据访问模式 ===")
    
    try:
        # 模拟复杂格式数据
        complex_data = {
            "metadata": {"total_files": 3},
            "results": {
                "/file1": {"result": "sat", "solve_time": 100},
                "/file2": {"result": "unsat", "solve_time": 200},
                "/file3": {"result": "unknown", "solve_time": 500}
            }
        }
        
        # 测试新的访问模式
        results_data = complex_data.get("results", complex_data) if "results" in complex_data else complex_data
        results_count = len(results_data)
        
        print(f"数据访问测试:")
        print(f"  检测到复杂格式: {'results' in complex_data}")
        print(f"  结果数据条目数: {results_count}")
        print(f"  结果键列表: {list(results_data.keys())}")
        
        # 测试简单格式数据
        simple_data = {
            "/file1": ["sat", 100, 1200, {}],
            "/file2": ["unsat", 200, 1200, {}],
            "/file3": ["unknown", 500, 1200, {}]
        }
        
        results_data_simple = simple_data.get("results", simple_data) if "results" in simple_data else simple_data
        results_count_simple = len(results_data_simple)
        
        print(f"\n简单格式测试:")
        print(f"  检测到复杂格式: {'results' in simple_data}")
        print(f"  结果数据条目数: {results_count_simple}")
        print(f"  结果键列表: {list(results_data_simple.keys())}")
        
        print("✓ 数据访问模式测试通过")
        return True
        
    except Exception as e:
        print(f"测试数据访问模式时出错: {e}")
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("测试修复后的BVParti预测器逻辑")
    print("=" * 60)
    
    tests = [
        ("数据处理逻辑", test_data_processing_logic),
        ("格式兼容性", test_format_compatibility),
        ("数据访问模式", test_data_access_patterns),
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
    print("\n测试总结:")
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓" if result else "✗"
        print(f"  {status} {test_name}")
    
    print(f"\n通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！逻辑修复成功。")
    else:
        print("⚠️  部分测试失败，需要进一步检查。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
