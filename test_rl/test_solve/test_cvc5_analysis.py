#!/usr/bin/env python3
"""
CVC5处理结果分析测试脚本
演示如何使用test_group_cvc5_process_analysis方法分析CVC5处理结果
"""

import sys
import os
sys.path.append('/home/lz/PycharmProjects/Pearl')

from test_rl.test_solve.test_solver_result import test_group_cvc5_process_analysis

def main():
    """主函数"""
    print("CVC5处理结果分析测试")
    print("=" * 60)

    # 文件路径 - 可以通过命令行参数或环境变量自定义
    info_name = '/home/lz/PycharmProjects/Pearl/test_rl/test_cvc5/cvc5_process/info_dict_SMTimer_llama3.1:70b_1200s_info_dict_rl_cvc5_0628.txt'
    var_count_path = '/home/lz/PycharmProjects/Pearl/test_rl/test_solve/cvc5_smtimer_var_count.txt'

    # 支持命令行参数
    if len(sys.argv) > 1:
        info_name = sys.argv[1]
    if len(sys.argv) > 2:
        var_count_path = sys.argv[2]
    
    # 检查文件是否存在
    if not os.path.exists(info_name):
        print(f"错误: 找不到info文件: {info_name}")
        return 1
    
    if not os.path.exists(var_count_path):
        print(f"错误: 找不到var_count文件: {var_count_path}")
        return 1
    
    print(f"Info文件: {info_name}")
    print(f"Var count文件: {var_count_path}")
    print()
    
    try:
        # 调用分析方法
        print("开始分析CVC5处理结果...")
        result_dict, time_dict, time_dict_2, info_dict = test_group_cvc5_process_analysis(info_name, var_count_path)
        
        print("\n" + "=" * 60)
        print("分析完成！")
        print("=" * 60)
        
        # 额外的统计信息
        print("\n详细统计信息:")
        print("-" * 40)
        
        total_succeed = len(result_dict['succeed'])
        total_failed = len(result_dict['failed'])
        total_files = total_succeed + total_failed
        
        if total_files > 0:
            success_rate = (total_succeed / total_files) * 100
            print(f"总文件数: {total_files}")
            print(f"成功数: {total_succeed}")
            print(f"失败数: {total_failed}")
            print(f"成功率: {success_rate:.2f}%")
        
        print("\n分类详情:")
        print("-" * 40)
        for category, files in result_dict.items():
            print(f"{category}: {len(files)} 个文件")
        
        # 时间统计
        positive_times = [t for t in time_dict.values() if t > 0]
        negative_times = [abs(t) for t in time_dict.values() if t < 0]
        
        if positive_times:
            print(f"\n原始求解器成功案例平均时间: {sum(positive_times)/len(positive_times):.2f}秒")
        if negative_times:
            print(f"原始求解器失败案例平均时间: {sum(negative_times)/len(negative_times):.2f}秒")
        
        positive_times_2 = [t for t in time_dict_2.values() if t > 0]
        negative_times_2 = [abs(t) for t in time_dict_2.values() if t < 0]
        
        if positive_times_2:
            print(f"RL+LLM成功案例平均时间: {sum(positive_times_2)/len(positive_times_2):.2f}秒")
        if negative_times_2:
            print(f"RL+LLM失败案例平均时间: {sum(negative_times_2)/len(negative_times_2):.2f}秒")
        
        return 0
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
