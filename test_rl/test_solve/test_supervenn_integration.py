#!/usr/bin/env python3
"""
测试SuperVenn集成功能的脚本
"""

import os
import sys
sys.path.append('/home/lz/PycharmProjects/Pearl')

from test_rl.test_solve.test_solver_result import test_group_cvc5_process_analysis

def test_supervenn_integration():
    """测试SuperVenn集成功能"""
    
    # 测试文件路径
    info_name = '/home/lz/PycharmProjects/Pearl/test_rl/test_cvc5/cvc5_process/info_dict_SMTimer_llama3.1:70b_1200s_info_dict_rl_cvc5_0628.txt'
    var_count_path = '/home/lz/PycharmProjects/Pearl/test_rl/test_solve/cvc5_smtimer_var_count.txt'
    output_dir = '/home/lz/PycharmProjects/Pearl/test_rl/test_cvc5/supervenn_test_output'
    
    # 检查输入文件是否存在
    if not os.path.exists(info_name):
        print(f"错误: 找不到信息文件 {info_name}")
        return False
    
    if not os.path.exists(var_count_path):
        print(f"错误: 找不到变量统计文件 {var_count_path}")
        return False
    
    print("开始测试SuperVenn集成功能...")
    print(f"信息文件: {info_name}")
    print(f"变量统计文件: {var_count_path}")
    print(f"输出目录: {output_dir}")
    print("-" * 80)
    
    try:
        # 调用增强的分析方法
        result_dict, time_dict, time_dict_2, info_dict, supervenn_stats = test_group_cvc5_process_analysis(
            info_name=info_name,
            var_count_path=var_count_path,
            output_dir=output_dir,
            solver_name="CVC5"
        )
        
        print("-" * 80)
        print("测试完成!")
        
        if supervenn_stats:
            print("\nSuperVenn统计信息:")
            for key, value in supervenn_stats.items():
                if isinstance(value, set):
                    print(f"  {key}: {len(value)} 个约束")
                else:
                    print(f"  {key}: {value}")
        
        # 检查输出文件是否生成
        expected_output = os.path.join(output_dir, "supervenn_cvc5_comparison.pdf")
        if os.path.exists(expected_output):
            print(f"\n✅ SuperVenn图已成功生成: {expected_output}")
        else:
            print(f"\n❌ SuperVenn图未生成: {expected_output}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_without_output_dir():
    """测试不生成SuperVenn图的情况"""
    
    info_name = '/home/lz/PycharmProjects/Pearl/test_rl/test_cvc5/cvc5_process/info_dict_SMTimer_llama3.1:70b_1200s_info_dict_rl_cvc5_0628.txt'
    var_count_path = '/home/lz/PycharmProjects/Pearl/test_rl/test_solve/cvc5_smtimer_var_count.txt'
    
    print("\n" + "="*80)
    print("测试不生成SuperVenn图的情况...")
    print("="*80)
    
    try:
        # 不指定output_dir
        result_dict, time_dict, time_dict_2, info_dict, supervenn_stats = test_group_cvc5_process_analysis(
            info_name=info_name,
            var_count_path=var_count_path,
            output_dir=None,  # 不生成图片
            solver_name="CVC5"
        )
        
        print("✅ 无输出目录测试成功")
        print(f"SuperVenn统计信息: {supervenn_stats}")
        return True
        
    except Exception as e:
        print(f"❌ 无输出目录测试失败: {e}")
        return False

if __name__ == "__main__":
    print("SuperVenn集成功能测试")
    print("=" * 80)
    
    # 测试1: 生成SuperVenn图
    success1 = test_supervenn_integration()
    
    # 测试2: 不生成SuperVenn图
    success2 = test_without_output_dir()
    
    print("\n" + "=" * 80)
    print("测试总结:")
    print(f"  生成SuperVenn图测试: {'✅ 通过' if success1 else '❌ 失败'}")
    print(f"  无输出目录测试: {'✅ 通过' if success2 else '❌ 失败'}")
    
    if success1 and success2:
        print("\n🎉 所有测试通过!")
        sys.exit(0)
    else:
        print("\n❌ 部分测试失败")
        sys.exit(1)
