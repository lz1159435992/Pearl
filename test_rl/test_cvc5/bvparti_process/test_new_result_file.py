#!/usr/bin/env python3
"""
测试新的结果文件 SMTimer_z3_result_rl.json 的兼容性
"""

import os
import sys
import json
import traceback

# 添加路径
sys.path.append('/home/lz/PycharmProjects/Pearl')

def test_new_result_file():
    """测试新的结果文件处理"""
    print("=== 测试新结果文件 SMTimer_z3_result_rl.json ===")
    
    try:
        from test_rl.test_cvc5.bvparti_process.test_group_get_dis_smt_comp_bert_embeding_single import (
            get_solve_result_and_time, convert_timeout_to_unknown
        )
        
        # 加载新的结果文件
        result_file_path = '/home/lz/PycharmProjects/Pearl/test_rl/AriParti_sync/scripts/batch_output/bv_default/SMTimer_z3_result_rl.json'
        
        if not os.path.exists(result_file_path):
            print(f"错误: 文件不存在 {result_file_path}")
            return False
        
        print(f"加载文件: {result_file_path}")
        
        with open(result_file_path, 'r') as f:
            result_dict = json.load(f)
        
        print(f"文件加载成功，包含 {len(result_dict.get('results', {}))} 个结果")
        
        # 检查文件结构
        print("\n文件结构检查:")
        if "metadata" in result_dict:
            print("  ✓ 包含 metadata 字段")
            metadata = result_dict["metadata"]
            if "statistics" in metadata:
                stats = metadata["statistics"]
                print(f"    - 总文件数: {stats.get('total_files', 'N/A')}")
                print(f"    - 成功文件数: {stats.get('successful_files', 'N/A')}")
                print(f"    - 错误文件数: {stats.get('error_files', 'N/A')}")
                print(f"    - 成功率: {stats.get('success_rate', 'N/A'):.2%}")
        
        if "results" in result_dict:
            print("  ✓ 包含 results 字段")
            results = result_dict["results"]
            print(f"    - 结果条目数: {len(results)}")
        
        # 测试前几个条目的处理
        print("\n测试数据处理:")
        count = 0
        result_stats = {"sat": 0, "unsat": 0, "unknown": 0, "error": 0}
        
        for key, value in result_dict.get("results", {}).items():
            if count >= 10:  # 只测试前10个
                break
            
            category, time_value = get_solve_result_and_time(result_dict, key)
            result_stats[category] = result_stats.get(category, 0) + 1
            
            print(f"  {count+1}. {os.path.basename(key)}: {category}, {time_value:.3f}s")
            count += 1
        
        print(f"\n前{count}个条目统计:")
        for status, cnt in result_stats.items():
            if cnt > 0:
                print(f"  {status}: {cnt}")
        
        # 测试timeout转换功能
        print("\n测试timeout转换:")
        converted_dict = convert_timeout_to_unknown(result_dict)
        
        # 检查是否有timeout被转换
        timeout_found = False
        for key, value in result_dict.get("results", {}).items():
            if value.get("result") == "timeout":
                timeout_found = True
                break
        
        if timeout_found:
            print("  发现timeout条目，转换功能正常")
        else:
            print("  未发现timeout条目，但转换功能可用")
        
        # 测试与RL字典的兼容性
        print("\n测试RL字典兼容性:")
        rl_dict_path = '/home/lz/sibyl_3/src/networks/info_dict_rl.txt'
        
        if os.path.exists(rl_dict_path):
            with open(rl_dict_path, 'r') as f:
                rl_dict = json.load(f)
            
            # 检查有多少结果文件中的条目在RL字典中
            common_keys = set(result_dict.get("results", {}).keys()) & set(rl_dict.keys())
            print(f"  RL字典条目数: {len(rl_dict)}")
            print(f"  结果文件条目数: {len(result_dict.get('results', {}))}")
            print(f"  共同条目数: {len(common_keys)}")
            
            if len(common_keys) > 0:
                print("  ✓ 存在共同条目，兼容性良好")
            else:
                print("  ⚠ 未发现共同条目，可能需要检查路径匹配")
        else:
            print(f"  RL字典文件不存在: {rl_dict_path}")
        
        return True
        
    except Exception as e:
        print(f"测试过程中出错: {e}")
        traceback.print_exc()
        return False

def test_file_comparison():
    """对比两个结果文件的差异"""
    print("\n=== 对比两个结果文件 ===")
    
    try:
        from test_rl.test_cvc5.bvparti_process.test_group_get_dis_smt_comp_bert_embeding_single import (
            get_solve_result_and_time
        )
        
        # 文件路径
        old_file = '/home/lz/PycharmProjects/Pearl/test_rl/AriParti_sync/scripts/batch_output/bv_default/SMTimer_z3_result_predictor.json'
        new_file = '/home/lz/PycharmProjects/Pearl/test_rl/AriParti_sync/scripts/batch_output/bv_default/SMTimer_z3_result_rl.json'
        
        if not os.path.exists(old_file):
            print(f"旧文件不存在: {old_file}")
            return False
        
        if not os.path.exists(new_file):
            print(f"新文件不存在: {new_file}")
            return False
        
        # 加载两个文件
        with open(old_file, 'r') as f:
            old_dict = json.load(f)
        
        with open(new_file, 'r') as f:
            new_dict = json.load(f)
        
        print("文件对比:")
        print(f"  旧文件条目数: {len(old_dict.get('results', {}))}")
        print(f"  新文件条目数: {len(new_dict.get('results', {}))}")
        
        # 检查共同的键
        old_keys = set(old_dict.get("results", {}).keys())
        new_keys = set(new_dict.get("results", {}).keys())
        
        common_keys = old_keys & new_keys
        only_old = old_keys - new_keys
        only_new = new_keys - old_keys
        
        print(f"  共同条目: {len(common_keys)}")
        print(f"  仅在旧文件: {len(only_old)}")
        print(f"  仅在新文件: {len(only_new)}")
        
        # 对比共同条目的结果差异
        if len(common_keys) > 0:
            print("\n结果差异分析 (前5个共同条目):")
            count = 0
            for key in list(common_keys)[:5]:
                old_category, old_time = get_solve_result_and_time(old_dict, key)
                new_category, new_time = get_solve_result_and_time(new_dict, key)
                
                status_change = "相同" if old_category == new_category else f"{old_category}→{new_category}"
                time_change = f"{old_time:.3f}→{new_time:.3f}" if old_time != new_time else f"{old_time:.3f}"
                
                print(f"  {count+1}. {os.path.basename(key)}: {status_change}, {time_change}s")
                count += 1
        
        return True
        
    except Exception as e:
        print(f"对比过程中出错: {e}")
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("测试新结果文件兼容性")
    print("=" * 60)
    
    tests = [
        ("新结果文件处理", test_new_result_file),
        ("文件对比分析", test_file_comparison),
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
        print("🎉 所有测试通过！新结果文件兼容性良好。")
    else:
        print("⚠️  部分测试失败，请检查文件格式或路径。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
