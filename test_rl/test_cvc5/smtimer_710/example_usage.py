#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SMT Solver CLI 使用示例
展示如何使用重构后的命令行工具
"""

import os
import subprocess
import sys

def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n{'='*60}")
    print(f"执行: {description}")
    print(f"命令: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        print("输出:")
        print(result.stdout)
        if result.stderr:
            print("错误:")
            print(result.stderr)
        print(f"返回码: {result.returncode}")
    except subprocess.TimeoutExpired:
        print("命令执行超时")
    except Exception as e:
        print(f"执行失败: {e}")

def main():
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cli_script = os.path.join(script_dir, 'smt_solver_cli_simple.py')
    
    print("SMT Solver CLI 使用示例")
    print("="*60)
    
    # 1. 显示帮助信息
    run_command([sys.executable, cli_script, '--help'], "显示总体帮助信息")
    
    # 2. 显示批量求解帮助
    run_command([sys.executable, cli_script, 'batch', '--help'], "显示批量求解帮助信息")
    
    # 3. 显示单文件求解帮助
    run_command([sys.executable, cli_script, 'single', '--help'], "显示单文件求解帮助信息")
    
    # 4. 运行测试
    run_command([sys.executable, cli_script, 'test'], "运行求解器可用性测试")
    
    # 5. 示例：使用CVC5求解器（如果有测试文件的话）
    test_file = os.path.join(script_dir, 'test_example.smt2')
    if os.path.exists(test_file):
        run_command([
            sys.executable, cli_script, 'single',
            '--file', test_file,
            '--solver', 'cvc5',
            '--timeout', '10'
        ], "求解单个测试文件")
    else:
        print("\n注意: 没有找到测试文件，跳过单文件求解示例")
        print("可以创建一个简单的SMT文件来测试单文件求解功能")
    
    # 6. 示例：批量求解（如果有信息文件的话）
    info_file = os.path.join(script_dir, 'smtimer-533-result.txt')
    if os.path.exists(info_file):
        print("\n注意: 找到信息文件，可以运行批量求解示例")
        print("示例命令:")
        print(f"python3 {cli_script} batch --info-file {info_file} --solver cvc5 --timeout 60 --max-workers 2")
    else:
        print("\n注意: 没有找到信息文件，跳过批量求解示例")
        print("可以创建一个包含文件路径的文本文件来测试批量求解功能")
    
    print("\n" + "="*60)
    print("示例演示完成！")
    print("="*60)

if __name__ == "__main__":
    main() 