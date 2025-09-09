#!/usr/bin/env python3
"""
演示如何使用var_count参数的脚本
展示test_group_2_no_save_1207和test_group_cvc5_process_analysis方法的参数使用
"""

import sys
import os
sys.path.append('/home/lz/PycharmProjects/Pearl')

from test_rl.test_solve.test_solver_result import test_group_2_no_save_1207, test_group_cvc5_process_analysis

def demo_test_group_2_no_save_1207():
    """演示test_group_2_no_save_1207方法的var_count参数使用"""
    print("=" * 80)
    print("演示 test_group_2_no_save_1207 方法")
    print("=" * 80)
    
    # 文件路径
    solve_name = '/home/lz/PycharmProjects/Pearl/test_rl/test_cvc5/cvc5_smtimer_results_rl.json'
    info_name = '/home/lz/PycharmProjects/Pearl/test_rl/test_cvc5/cvc5_process/info_dict_SMTimer_llama3.1:70b_1200s_info_dict_rl_cvc5_0628.txt'
    
    # 方法1: 使用默认var_count路径
    print("方法1: 使用默认var_count路径")
    print("-" * 40)
    try:
        result_dict_1, time_dict_1, time_dict_2_1, info_dict_1 = test_group_2_no_save_1207(solve_name, info_name)
        print("✅ 使用默认路径成功")
    except Exception as e:
        print(f"❌ 使用默认路径失败: {e}")
    
    print()
    
    # 方法2: 使用自定义var_count路径
    print("方法2: 使用自定义var_count路径")
    print("-" * 40)
    custom_var_count_path = '/home/lz/PycharmProjects/Pearl/test_rl/test_solve/cvc5_smtimer_var_count.txt'
    try:
        result_dict_2, time_dict_2, time_dict_2_2, info_dict_2 = test_group_2_no_save_1207(
            solve_name, info_name, custom_var_count_path
        )
        print("✅ 使用自定义路径成功")
        print(f"使用的var_count路径: {custom_var_count_path}")
    except Exception as e:
        print(f"❌ 使用自定义路径失败: {e}")

def demo_test_group_cvc5_process_analysis():
    """演示test_group_cvc5_process_analysis方法的var_count参数使用"""
    print("\n" + "=" * 80)
    print("演示 test_group_cvc5_process_analysis 方法")
    print("=" * 80)
    
    # 文件路径
    info_name = '/home/lz/PycharmProjects/Pearl/test_rl/test_cvc5/cvc5_process/info_dict_SMTimer_llama3.1:70b_1200s_info_dict_rl_cvc5_0628.txt'
    
    # 方法1: 使用默认var_count路径
    print("方法1: 使用默认var_count路径")
    print("-" * 40)
    try:
        result_dict_1, time_dict_1, time_dict_2_1, info_dict_1 = test_group_cvc5_process_analysis(info_name)
        print("✅ 使用默认路径成功")
        print(f"处理的文件数: {len(result_dict_1['succeed']) + len(result_dict_1['failed'])}")
    except Exception as e:
        print(f"❌ 使用默认路径失败: {e}")
    
    print()
    
    # 方法2: 使用自定义var_count路径
    print("方法2: 使用自定义var_count路径")
    print("-" * 40)
    custom_var_count_path = '/home/lz/PycharmProjects/Pearl/test_rl/test_solve/cvc5_smtimer_var_count.txt'
    try:
        result_dict_2, time_dict_2, time_dict_2_2, info_dict_2 = test_group_cvc5_process_analysis(
            info_name, custom_var_count_path
        )
        print("✅ 使用自定义路径成功")
        print(f"使用的var_count路径: {custom_var_count_path}")
        print(f"处理的文件数: {len(result_dict_2['succeed']) + len(result_dict_2['failed'])}")
    except Exception as e:
        print(f"❌ 使用自定义路径失败: {e}")

def compare_methods():
    """比较两种方法的结果"""
    print("\n" + "=" * 80)
    print("比较两种方法的参数使用方式")
    print("=" * 80)
    
    print("方法签名对比:")
    print("-" * 40)
    print("test_group_2_no_save_1207(solve_name, info_name, var_count_path=默认路径)")
    print("test_group_cvc5_process_analysis(info_name, var_count_path=默认路径)")
    print()
    
    print("参数说明:")
    print("-" * 40)
    print("• solve_name: 传统求解器结果文件 (仅test_group_2_no_save_1207需要)")
    print("• info_name: RL+LLM处理结果文件 (两个方法都需要)")
    print("• var_count_path: 变量统计文件 (两个方法都支持，都有默认值)")
    print()
    
    print("使用场景:")
    print("-" * 40)
    print("• test_group_2_no_save_1207: 需要对比传统求解器和RL+LLM的结果")
    print("• test_group_cvc5_process_analysis: 只分析RL+LLM的处理结果")
    print()
    
    print("var_count参数的作用:")
    print("-" * 40)
    print("• 用于筛选变量数量 > 5 的测试用例")
    print("• 支持使用不同的变量统计文件")
    print("• 默认路径: /home/lz/PycharmProjects/Pearl/test_rl/test_solve/var_count.txt")
    print("• 可自定义路径，如: cvc5_smtimer_var_count.txt")

def main():
    """主函数"""
    print("var_count参数使用演示")
    print("展示如何在两个分析方法中使用var_count参数")
    
    # 检查必要文件是否存在
    required_files = [
        '/home/lz/PycharmProjects/Pearl/test_rl/test_cvc5/cvc5_smtimer_results_rl.json',
        '/home/lz/PycharmProjects/Pearl/test_rl/test_cvc5/cvc5_process/info_dict_SMTimer_llama3.1:70b_1200s_info_dict_rl_cvc5_0628.txt',
        '/home/lz/PycharmProjects/Pearl/test_rl/test_solve/cvc5_smtimer_var_count.txt'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("❌ 以下必需文件不存在:")
        for f in missing_files:
            print(f"   {f}")
        return 1
    
    try:
        # 演示test_group_2_no_save_1207
        demo_test_group_2_no_save_1207()
        
        # 演示test_group_cvc5_process_analysis
        demo_test_group_cvc5_process_analysis()
        
        # 比较两种方法
        compare_methods()
        
        print("\n" + "=" * 80)
        print("✅ 演示完成！两个方法都支持var_count参数")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
