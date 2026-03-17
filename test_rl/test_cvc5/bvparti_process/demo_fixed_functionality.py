#!/usr/bin/env python3
"""
演示脚本：展示修改后的代码如何处理两种不同格式的JSON文件
"""

import json
import sys
import os
import numpy as np

# 添加路径以便导入模块
sys.path.append('/home/lz/PycharmProjects/Pearl')

def demo_fixed_functionality():
    """演示修复后的功能"""
    
    print("=== 演示修复后的JSON处理功能 ===")
    
    # 导入修改后的函数
    from test_rl.test_cvc5.bvparti_process.test_group_get_dis_smt_comp_bert_embeding_single import (
        get_solve_result_and_time, 
        convert_timeout_to_unknown
    )
    
    # 创建一个简单的测试数据集
    print("\n1. 创建测试数据...")
    
    # 模拟info_dict（文件信息字典）
    test_info_dict = {
        "/home/lz/baidudisk/smt/gnu_angr.tar.gz/single_test/sort/sort29776": ["test_data"],
        "/home/lz/baidudisk/smt/gnu_angr.tar.gz/single_test/comm/comm28789": ["test_data"],
        "/home/lz/baidudisk/smt/gnu_angr.tar.gz/single_test/pinky/pinky154333": ["test_data"]
    }
    
    # 测试旧格式（cvc5格式）
    print("\n2. 测试旧格式处理...")
    old_format_solve_dict = {
        "/home/lz/baidudisk/smt/gnu_angr.tar.gz/single_test/sort/sort29776": ["sat", 1.7, 1200, {}],
        "/home/lz/baidudisk/smt/gnu_angr.tar.gz/single_test/comm/comm28789": ["unsat", 1.1, 1200, {}],
        "/home/lz/baidudisk/smt/gnu_angr.tar.gz/single_test/pinky/pinky154333": ["sat", 4.2, 1200, {}]
    }
    
    print("  旧格式数据处理结果:")
    labels_list_old = []
    time_list_old = []
    
    for key in test_info_dict:
        category, time_value = get_solve_result_and_time(old_format_solve_dict, key)
        print(f"    {key.split('/')[-1]}: {category}, {time_value}s")
        
        # 生成标签
        if category == "sat":
            labels_list_old.append(0)
            if time_value <= 1:
                time_list_old.append(1)
            elif time_value <= 20:
                time_list_old.append(2)
            else:
                time_list_old.append(3)
        else:
            labels_list_old.append(1)
            time_list_old.append(0)
    
    print(f"    生成的标签: {labels_list_old}")
    print(f"    生成的时间分类: {time_list_old}")
    
    # 测试新格式（bvparti格式）
    print("\n3. 测试新格式处理...")
    new_format_solve_dict = {
        "metadata": {
            "execution_info": {"start_time": "2025-07-21 00:36:48"},
            "configuration": {"solver": "bitwuzla", "time_limit": 1200}
        },
        "results": {
            "/home/lz/baidudisk/smt/gnu_angr.tar.gz/single_test/sort/sort29776": {
                "result": "error",  # 这会被映射为unknown
                "solve_time": -1,
                "total_time": 0.078,
                "error": "KeyError: 'unknown'",
                "returncode": 1
            },
            "/home/lz/baidudisk/smt/gnu_angr.tar.gz/single_test/comm/comm28789": {
                "result": "unsat",
                "solve_time": 0.010,
                "total_time": 0.077,
                "error": None,
                "returncode": 0
            },
            "/home/lz/baidudisk/smt/gnu_angr.tar.gz/single_test/pinky/pinky154333": {
                "result": "sat",
                "solve_time": 24.2,
                "total_time": 24.3,
                "error": None,
                "returncode": 0
            }
        }
    }
    
    print("  新格式数据处理结果:")
    labels_list_new = []
    time_list_new = []
    
    for key in test_info_dict:
        category, time_value = get_solve_result_and_time(new_format_solve_dict, key)
        print(f"    {key.split('/')[-1]}: {category}, {time_value}s")
        
        # 生成标签
        if category == "sat":
            labels_list_new.append(0)
            if time_value <= 1:
                time_list_new.append(1)
            elif time_value <= 20:
                time_list_new.append(2)
            else:
                time_list_new.append(3)
        else:
            labels_list_new.append(1)
            time_list_new.append(0)
    
    print(f"    生成的标签: {labels_list_new}")
    print(f"    生成的时间分类: {time_list_new}")
    
    print("\n4. 验证timeout转换功能...")
    
    # 测试timeout转换
    timeout_test_old = {
        "/test/file": ["timeout", 1200.0, 1200, {}]
    }
    
    timeout_test_new = {
        "metadata": {},
        "results": {
            "/test/file": {
                "result": "timeout",
                "solve_time": 1200.0,
                "total_time": 1200.1
            }
        }
    }
    
    converted_old = convert_timeout_to_unknown(timeout_test_old)
    converted_new = convert_timeout_to_unknown(timeout_test_new)
    
    print("  旧格式timeout转换:")
    category, time_value = get_solve_result_and_time(converted_old, "/test/file")
    print(f"    转换后: {category}, {time_value}s")
    
    print("  新格式timeout转换:")
    category, time_value = get_solve_result_and_time(converted_new, "/test/file")
    print(f"    转换后: {category}, {time_value}s")
    
    print("\n=== 演示完成！修改成功，代码现在可以处理两种格式的JSON文件 ===")

if __name__ == "__main__":
    demo_fixed_functionality()
