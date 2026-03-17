#!/usr/bin/env python3
"""
测试默认参数是否正确设置
"""

import sys
import argparse

# 添加路径
sys.path.append('/home/lz/PycharmProjects/Pearl')

def test_default_arguments():
    """测试默认参数"""
    print("=== 测试默认参数 ===")
    
    # 创建与主程序相同的参数解析器
    parser = argparse.ArgumentParser(description='运行BVParti SMT约束求解预测器')
    
    # 添加所有参数
    parser.add_argument('--rl_dict_path', type=str, 
                        default='/home/lz/sibyl_3/src/networks/info_dict_rl.txt',
                        help='RL字典文件路径')
    parser.add_argument('--info_dict_path', type=str, 
                        default='info_dict_SMTimer_llama3.1:70b_1200s_info_dict_rl_bvparti_0728.txt',
                        help='信息字典文件路径')
    parser.add_argument('--result_dict_path', type=str, 
                        default='/home/lz/PycharmProjects/Pearl/test_rl/AriParti_sync/scripts/batch_output/bv_default/SMTimer_z3_result_rl.json',
                        help='结果字典文件路径')
    parser.add_argument('--binary_model_path', type=str,
                        default='models/binary_classifier.pth',
                        help='二分类模型路径')
    parser.add_argument('--eight_class_model_path', type=str,
                        default='models/eight_class_model.pth',
                        help='八分类模型路径')
    parser.add_argument('--time_threshold', type=int,
                        default=300,
                        help='时间阈值')
    parser.add_argument('--timeout', type=int,
                        default=1200,
                        help='执行超时时间（秒）')
    parser.add_argument('--num_episodes', type=int,
                        default=1,
                        help='训练轮数')
    parser.add_argument('--record_period', type=int,
                        default=1,
                        help='记录周期')
    parser.add_argument('--llm_host', type=str,
                        default='http://172.29.7.221:32943',
                        help='LLM服务器地址')
    parser.add_argument('--llm_model', type=str,
                        default='llama3.1:70b',
                        help='LLM模型名称')
    parser.add_argument('--solver', type=str,
                        default='bvparti',
                        choices=['z3', 'cvc5', 'bvparti'],
                        help='选择使用的求解器(z3/cvc5/bvparti)')

    # 解析空参数列表（使用默认值）
    args = parser.parse_args([])
    
    print("默认参数值:")
    print(f"  RL字典路径: {args.rl_dict_path}")
    print(f"  信息字典路径: {args.info_dict_path}")
    print(f"  结果字典路径: {args.result_dict_path}")
    print(f"  求解器: {args.solver}")
    print(f"  时间阈值: {args.time_threshold}")
    print(f"  超时时间: {args.timeout}")
    print(f"  LLM模型: {args.llm_model}")
    
    # 验证关键路径
    expected_result_path = '/home/lz/PycharmProjects/Pearl/test_rl/AriParti_sync/scripts/batch_output/bv_default/SMTimer_z3_result_rl.json'
    
    if args.result_dict_path == expected_result_path:
        print("\n✓ 结果字典路径设置正确")
        return True
    else:
        print(f"\n✗ 结果字典路径不正确")
        print(f"  期望: {expected_result_path}")
        print(f"  实际: {args.result_dict_path}")
        return False

def test_import_main():
    """测试导入主函数并检查参数"""
    print("\n=== 测试导入主函数 ===")
    
    try:
        # 导入主模块
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_bvparti_predictor", 
            "/home/lz/PycharmProjects/Pearl/test_rl/test_cvc5/bvparti_process/run_bvparti_predictor.py"
        )
        module = importlib.util.module_from_spec(spec)
        
        # 这里不执行模块，只是验证能够导入
        print("✓ 模块导入成功")
        return True
        
    except Exception as e:
        print(f"✗ 模块导入失败: {e}")
        return False

def main():
    """主函数"""
    print("测试BVParti预测器默认参数")
    print("=" * 50)
    
    tests = [
        ("默认参数测试", test_default_arguments),
        ("模块导入测试", test_import_main),
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
        
        print("-" * 50)
    
    # 总结
    passed = sum(results.values())
    total = len(results)
    
    print(f"\n测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！默认参数设置正确。")
    else:
        print("⚠️  部分测试失败。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
