#!/usr/bin/env python3
import json
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

def analyze_smt_results(file_path: str) -> Tuple[Dict[str, int], Dict[str, float], Dict[str, float]]:
    """
    分析SMT求解结果文件，统计各类结果的数量和时间
    
    Args:
        file_path: JSON结果文件的路径
        
    Returns:
        counts: 每种结果的数量统计
        total_times: 每种结果的总时间
        avg_times: 每种结果的平均时间
    """
    with open(file_path, 'r') as f:
        results = json.load(f)
    
    # 初始化统计数据
    counts = defaultdict(int)
    total_times = defaultdict(float)
    
    # 统计每种结果的数量和时间
    for test_case, result in results.items():
        status = result[0]  # sat/unsat/unknown
        time = result[1]    # 求解时间
        
        counts[status] += 1
        total_times[status] += time
    
    # 计算平均时间
    avg_times = {
        status: total_times[status] / count 
        for status, count in counts.items()
    }
    
    return dict(counts), dict(total_times), dict(avg_times)

def main():
    if len(sys.argv) != 2:
        print("使用方法: python3 analyze_smt_results.py <结果文件路径>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    counts, total_times, avg_times = analyze_smt_results(file_path)
    
    # 打印统计结果
    print("\n=== SMT求解结果统计 ===")
    print("\n各状态数量统计:")
    for status, count in counts.items():
        print(f"{status}: {count}次")
    
    print("\n各状态总时间统计:")
    for status, total_time in total_times.items():
        print(f"{status}: {total_time:.2f}秒")
    
    print("\n各状态平均时间统计:")
    for status, avg_time in avg_times.items():
        print(f"{status}: {avg_time:.2f}秒")

if __name__ == "__main__":
    main() 