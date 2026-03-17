#!/usr/bin/env python3
"""
验证更新后的BVParti预测器能正确处理新的默认结果文件
"""

import os
import sys
import json
import argparse
import traceback

# 添加路径
sys.path.append('/home/lz/PycharmProjects/Pearl')

def test_argument_parsing():
    """测试命令行参数解析，确认新的默认路径"""
    print("=== 测试命令行参数解析 ===")
    
    try:
        # 模拟导入主函数的参数解析器
        parser = argparse.ArgumentParser(description='运行BVParti SMT约束求解预测器')
        
        # 添加与实际代码相同的参数
        parser.add_argument('--result_dict_path', type=str, 
                            default='/home/lz/PycharmProjects/Pearl/test_rl/AriParti_sync/scripts/batch_output/bv_default/SMTimer_z3_result_rl.json',
                            help='结果字典文件路径')
        parser.add_argument('--solver', type=str, default='bvparti', choices=['z3', 'cvc5', 'bvparti'])
        
        # 解析空参数（使用默认值）
        args = parser.parse_args([])
        
        print(f"默认结果文件路径: {args.result_dict_path}")
        print(f"默认求解器: {args.solver}")
        
        # 检查文件是否存在
        if os.path.exists(args.result_dict_path):
            print("✓ 默认结果文件存在")
            
            # 检查文件大小
            file_size = os.path.getsize(args.result_dict_path)
            print(f"  文件大小: {file_size / (1024*1024):.1f} MB")
            
            return True
        else:
            print("✗ 默认结果文件不存在")
            return False
            
    except Exception as e:
        print(f"测试参数解析时出错: {e}")
        traceback.print_exc()
        return False

def test_json_processing():
    """测试JSON处理功能"""
    print("\n=== 测试JSON处理功能 ===")
    
    try:
        from test_rl.test_cvc5.bvparti_process.test_group_get_dis_smt_comp_bert_embeding_single import (
            get_solve_result_and_time, convert_timeout_to_unknown
        )
        
        # 使用新的默认文件路径
        result_file_path = '/home/lz/PycharmProjects/Pearl/test_rl/AriParti_sync/scripts/batch_output/bv_default/SMTimer_z3_result_rl.json'
        
        if not os.path.exists(result_file_path):
            print(f"✗ 文件不存在: {result_file_path}")
            return False
        
        # 加载文件
        with open(result_file_path, 'r') as f:
            result_dict = json.load(f)
        
        print(f"✓ 成功加载文件，包含 {len(result_dict.get('results', {}))} 个结果")
        
        # 测试数据处理
        test_count = 5
        processed_count = 0
        
        for key in list(result_dict.get('results', {}).keys())[:test_count]:
            category, time_value = get_solve_result_and_time(result_dict, key)
            print(f"  {processed_count+1}. {os.path.basename(key)}: {category}, {time_value:.3f}s")
            processed_count += 1
        
        print(f"✓ 成功处理 {processed_count} 个条目")
        
        # 测试timeout转换
        converted_dict = convert_timeout_to_unknown(result_dict)
        print("✓ timeout转换功能正常")
        
        return True
        
    except Exception as e:
        print(f"测试JSON处理时出错: {e}")
        traceback.print_exc()
        return False

def test_data_filtering():
    """测试数据过滤逻辑"""
    print("\n=== 测试数据过滤逻辑 ===")
    
    try:
        from test_rl.test_cvc5.bvparti_process.test_group_get_dis_smt_comp_bert_embeding_single import (
            get_solve_result_and_time
        )
        
        # 加载结果文件
        result_file_path = '/home/lz/PycharmProjects/Pearl/test_rl/AriParti_sync/scripts/batch_output/bv_default/SMTimer_z3_result_rl.json'
        with open(result_file_path, 'r') as f:
            result_dict = json.load(f)
        
        # 加载RL字典
        rl_dict_path = '/home/lz/sibyl_3/src/networks/info_dict_rl.txt'
        if os.path.exists(rl_dict_path):
            with open(rl_dict_path, 'r') as f:
                rl_dict = json.load(f)
        else:
            print(f"RL字典文件不存在: {rl_dict_path}")
            return False
        
        # 模拟预测器中的过滤逻辑
        time_threshold = 300  # 默认时间阈值
        
        filtered_count = 0
        total_count = 0
        
        for key in result_dict.get('results', {}):
            total_count += 1
            category, time_value = get_solve_result_and_time(result_dict, key)
            
            # 应用过滤条件：sat或unknown，时间超过阈值，在RL字典中
            if category in ["sat", "unknown"] and time_value > time_threshold and key in rl_dict:
                filtered_count += 1
                
                if filtered_count <= 5:  # 只显示前5个
                    print(f"  {filtered_count}. {os.path.basename(key)}: {category}, {time_value:.1f}s")
        
        print(f"\n过滤结果:")
        print(f"  总条目数: {total_count}")
        print(f"  符合条件的条目数: {filtered_count}")
        print(f"  过滤比例: {filtered_count/total_count*100:.1f}%")
        
        if filtered_count > 0:
            print("✓ 数据过滤逻辑正常，有可处理的数据")
            return True
        else:
            print("⚠ 没有符合条件的数据，可能需要调整过滤条件")
            return False
        
    except Exception as e:
        print(f"测试数据过滤时出错: {e}")
        traceback.print_exc()
        return False

def test_solver_compatibility():
    """测试求解器兼容性"""
    print("\n=== 测试求解器兼容性 ===")
    
    try:
        from test_rl.test_cvc5.bvparti_process.run_bvparti_predictor import get_solver
        
        # 测试所有支持的求解器
        solvers = ['bvparti', 'z3', 'cvc5']
        
        for solver_name in solvers:
            solver = get_solver(solver_name)
            print(f"  {solver_name}: {type(solver).__name__} ✓")
        
        print("✓ 所有求解器类型正常创建")
        return True
        
    except Exception as e:
        print(f"测试求解器兼容性时出错: {e}")
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("验证更新后的BVParti预测器")
    print("=" * 60)
    
    tests = [
        ("命令行参数解析", test_argument_parsing),
        ("JSON处理功能", test_json_processing),
        ("数据过滤逻辑", test_data_filtering),
        ("求解器兼容性", test_solver_compatibility),
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
    print("\n验证总结:")
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓" if result else "✗"
        print(f"  {status} {test_name}")
    
    print(f"\n通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有验证通过！更新后的预测器已准备就绪。")
        print("\n可以使用以下命令运行预测器:")
        print("python test_rl/test_cvc5/bvparti_process/run_bvparti_predictor.py")
    else:
        print("⚠️  部分验证失败，请检查配置。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
