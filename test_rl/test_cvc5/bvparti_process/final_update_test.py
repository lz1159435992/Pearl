#!/usr/bin/env python3
"""
最终更新测试：验证BVParti预测器使用新的默认结果文件路径
"""

import os
import sys
import json
import subprocess
import traceback

# 添加路径
sys.path.append('/home/lz/PycharmProjects/Pearl')

def test_help_command():
    """测试help命令，确认默认参数"""
    print("=== 测试help命令 ===")
    
    try:
        cmd = [
            sys.executable, 
            'test_rl/test_cvc5/bvparti_process/run_bvparti_predictor.py',
            '--help'
        ]
        
        result = subprocess.run(
            cmd,
            cwd='/home/lz/PycharmProjects/Pearl',
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            help_output = result.stdout
            
            # 检查默认路径是否正确
            if 'SMTimer_z3_result_rl.json' in help_output:
                print("✓ help输出包含正确的默认结果文件路径")
                return True
            else:
                print("✗ help输出中未找到正确的默认路径")
                print("Help输出片段:")
                lines = help_output.split('\n')
                for line in lines:
                    if 'result_dict_path' in line or 'SMTimer' in line:
                        print(f"  {line}")
                return False
        else:
            print(f"✗ help命令执行失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"测试help命令时出错: {e}")
        traceback.print_exc()
        return False

def test_file_access():
    """测试文件访问权限"""
    print("\n=== 测试文件访问权限 ===")
    
    files_to_check = [
        '/home/lz/PycharmProjects/Pearl/test_rl/AriParti_sync/scripts/batch_output/bv_default/SMTimer_z3_result_rl.json',
        '/home/lz/sibyl_3/src/networks/info_dict_rl.txt',
    ]
    
    all_accessible = True
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            if os.access(file_path, os.R_OK):
                file_size = os.path.getsize(file_path) / (1024*1024)
                print(f"✓ {os.path.basename(file_path)}: 可读，大小 {file_size:.1f} MB")
            else:
                print(f"✗ {os.path.basename(file_path)}: 存在但不可读")
                all_accessible = False
        else:
            print(f"✗ {os.path.basename(file_path)}: 不存在")
            all_accessible = False
    
    return all_accessible

def test_data_compatibility():
    """测试数据兼容性"""
    print("\n=== 测试数据兼容性 ===")
    
    try:
        from test_rl.test_cvc5.bvparti_process.test_group_get_dis_smt_comp_bert_embeding_single import (
            get_solve_result_and_time
        )
        
        # 加载新的结果文件
        result_file = '/home/lz/PycharmProjects/Pearl/test_rl/AriParti_sync/scripts/batch_output/bv_default/SMTimer_z3_result_rl.json'
        with open(result_file, 'r') as f:
            result_dict = json.load(f)
        
        # 加载RL字典
        rl_file = '/home/lz/sibyl_3/src/networks/info_dict_rl.txt'
        with open(rl_file, 'r') as f:
            rl_dict = json.load(f)
        
        # 统计兼容性
        result_keys = set(result_dict.get('results', {}).keys())
        rl_keys = set(rl_dict.keys())
        
        common_keys = result_keys & rl_keys
        coverage = len(common_keys) / len(rl_keys) * 100
        
        print(f"结果文件条目: {len(result_keys)}")
        print(f"RL字典条目: {len(rl_keys)}")
        print(f"共同条目: {len(common_keys)}")
        print(f"覆盖率: {coverage:.1f}%")
        
        if coverage > 95:
            print("✓ 数据兼容性优秀")
            return True
        elif coverage > 80:
            print("✓ 数据兼容性良好")
            return True
        else:
            print("⚠ 数据兼容性需要改进")
            return False
            
    except Exception as e:
        print(f"测试数据兼容性时出错: {e}")
        traceback.print_exc()
        return False

def test_filtering_logic():
    """测试过滤逻辑"""
    print("\n=== 测试过滤逻辑 ===")
    
    try:
        from test_rl.test_cvc5.bvparti_process.test_group_get_dis_smt_comp_bert_embeding_single import (
            get_solve_result_and_time
        )
        
        # 加载数据
        result_file = '/home/lz/PycharmProjects/Pearl/test_rl/AriParti_sync/scripts/batch_output/bv_default/SMTimer_z3_result_rl.json'
        with open(result_file, 'r') as f:
            result_dict = json.load(f)
        
        rl_file = '/home/lz/sibyl_3/src/networks/info_dict_rl.txt'
        with open(rl_file, 'r') as f:
            rl_dict = json.load(f)
        
        # 应用不同的时间阈值进行过滤
        thresholds = [100, 200, 300, 500, 1000]
        
        print("不同时间阈值下的可处理数据量:")
        
        for threshold in thresholds:
            count = 0
            for key in result_dict.get('results', {}):
                category, time_value = get_solve_result_and_time(result_dict, key)
                if category in ["sat", "unknown"] and time_value > threshold and key in rl_dict:
                    count += 1
            
            print(f"  {threshold}s阈值: {count} 个文件")
        
        # 检查是否有足够的数据进行处理
        count_300 = 0
        for key in result_dict.get('results', {}):
            category, time_value = get_solve_result_and_time(result_dict, key)
            if category in ["sat", "unknown"] and time_value > 300 and key in rl_dict:
                count_300 += 1
        
        if count_300 > 0:
            print(f"✓ 在300s阈值下有 {count_300} 个可处理文件")
            return True
        else:
            print("⚠ 在300s阈值下没有可处理文件")
            return False
            
    except Exception as e:
        print(f"测试过滤逻辑时出错: {e}")
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("BVParti预测器最终更新测试")
    print("=" * 60)
    print("验证使用新的默认结果文件路径: SMTimer_z3_result_rl.json")
    print("=" * 60)
    
    tests = [
        ("Help命令测试", test_help_command),
        ("文件访问权限", test_file_access),
        ("数据兼容性", test_data_compatibility),
        ("过滤逻辑", test_filtering_logic),
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
    print("\n最终测试总结:")
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓" if result else "✗"
        print(f"  {status} {test_name}")
    
    print(f"\n通过: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 所有测试通过！")
        print("✅ BVParti预测器已成功更新为使用新的默认结果文件")
        print("✅ 新文件路径: SMTimer_z3_result_rl.json")
        print("✅ JSON处理逻辑完全兼容")
        print("✅ 数据过滤和处理逻辑正常")
        print("\n现在可以直接运行预测器:")
        print("python test_rl/test_cvc5/bvparti_process/run_bvparti_predictor.py")
    else:
        print("\n⚠️  部分测试失败，请检查配置。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
