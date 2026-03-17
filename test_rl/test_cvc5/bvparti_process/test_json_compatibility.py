#!/usr/bin/env python3
"""
测试脚本：验证修改后的代码能否正确处理两种不同格式的JSON文件
"""

import json
import sys
import os

# 添加路径以便导入模块
sys.path.append('/home/lz/PycharmProjects/Pearl')

def test_json_formats():
    """测试两种JSON格式的兼容性"""
    
    # 导入修改后的函数
    from test_rl.test_cvc5.bvparti_process.test_group_get_dis_smt_comp_bert_embeding_single import (
        get_solve_result_and_time, 
        convert_timeout_to_unknown
    )
    
    print("=== 测试JSON格式兼容性 ===")
    
    # 测试旧格式（cvc5格式）
    print("\n1. 测试旧格式（cvc5格式）...")
    old_format = {
        "/path/to/file1": ["sat", 1.5, 1200, {}],
        "/path/to/file2": ["unsat", 0.8, 1200, {}],
        "/path/to/file3": ["timeout", 1200.0, 1200, {}]
    }
    
    # 转换timeout为unknown
    old_format_converted = convert_timeout_to_unknown(old_format)
    
    for key in old_format_converted:
        category, time_value = get_solve_result_and_time(old_format_converted, key)
        print(f"  {key}: {category}, {time_value}")
    
    # 测试新格式（bvparti格式）
    print("\n2. 测试新格式（bvparti格式）...")
    new_format = {
        "metadata": {
            "execution_info": {"start_time": "2025-07-21 00:36:48"},
            "configuration": {"solver": "bitwuzla", "time_limit": 1200},
            "statistics": {"total_files": 100}
        },
        "results": {
            "/path/to/file1": {
                "result": "sat",
                "solve_time": 1.5,
                "total_time": 1.6,
                "error": None,
                "returncode": 0
            },
            "/path/to/file2": {
                "result": "unsat", 
                "solve_time": 0.8,
                "total_time": 0.9,
                "error": None,
                "returncode": 0
            },
            "/path/to/file3": {
                "result": "error",
                "solve_time": -1,
                "total_time": 0.1,
                "error": "KeyError: 'unknown'",
                "returncode": 1
            },
            "/path/to/file4": {
                "result": "timeout",
                "solve_time": 1200.0,
                "total_time": 1200.1,
                "error": None,
                "returncode": 0
            }
        }
    }
    
    # 转换timeout为unknown
    new_format_converted = convert_timeout_to_unknown(new_format)
    
    for key in new_format_converted["results"]:
        category, time_value = get_solve_result_and_time(new_format_converted, key)
        print(f"  {key}: {category}, {time_value}")
    
    print("\n=== 测试完成 ===")
    
    # 验证实际文件
    print("\n3. 验证实际文件...")
    
    # 检查cvc5文件
    cvc5_file = "/home/lz/PycharmProjects/Pearl/test_rl/test_cvc5/cvc5_smtimer_results_predictor.json"
    if os.path.exists(cvc5_file):
        print(f"  检查cvc5文件: {cvc5_file}")
        with open(cvc5_file, 'r') as f:
            cvc5_data = json.load(f)
        
        # 取前3个条目测试
        count = 0
        for key, value in cvc5_data.items():
            if count >= 3:
                break
            category, time_value = get_solve_result_and_time(cvc5_data, key)
            print(f"    {key}: {category}, {time_value}")
            count += 1
    else:
        print(f"  cvc5文件不存在: {cvc5_file}")
    
    # 检查bvparti文件
    bvparti_file = "/home/lz/PycharmProjects/Pearl/test_rl/AriParti_sync/scripts/batch_output/bv_default/SMTimer_z3_result_predictor.json"
    if os.path.exists(bvparti_file):
        print(f"  检查bvparti文件: {bvparti_file}")
        with open(bvparti_file, 'r') as f:
            bvparti_data = json.load(f)
        
        # 取前3个条目测试
        if "results" in bvparti_data:
            count = 0
            for key, value in bvparti_data["results"].items():
                if count >= 3:
                    break
                category, time_value = get_solve_result_and_time(bvparti_data, key)
                print(f"    {key}: {category}, {time_value}")
                count += 1
        else:
            print("    bvparti文件格式不正确，缺少results字段")
    else:
        print(f"  bvparti文件不存在: {bvparti_file}")

if __name__ == "__main__":
    test_json_formats()
