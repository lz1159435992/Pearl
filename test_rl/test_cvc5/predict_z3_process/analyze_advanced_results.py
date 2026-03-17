#!/usr/bin/env python3
"""
分析advanced_solver_results_all.json文件
统计求解结果和时间分布
"""

import json
import os
import sys
from collections import defaultdict
import numpy as np

def load_json_file(file_path):
    """加载JSON文件"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def analyze_solver_results(data):
    """分析求解器结果"""
    print("=" * 80)
    print("Advanced Solver Results Analysis")
    print("=" * 80)
    
    # 统计变量
    total_constraints = len(data)
    solved_constraints = 0
    unsolved_constraints = 0
    sat_results = 0
    unsat_results = 0
    unknown_results = 0
    error_results = 0
    
    # 时间分布统计
    time_distribution = defaultdict(int)
    all_solve_times = []
    
    # 方法统计
    method_stats = defaultdict(int)
    
    # 状态统计
    status_stats = defaultdict(int)
    
    print(f"Total constraints: {total_constraints}")
    print("-" * 80)
    
    for constraint_path, result in data.items():
        if isinstance(result, list) and len(result) >= 2:
            # 提取结果信息
            result_status = result[0] if len(result) > 0 else 'unknown'
            solve_time = result[1] if len(result) > 1 else 0
            method_status = result[7] if len(result) > 7 else 'unknown'
            
            # 统计求解状态
            if result_status == 'sat':
                sat_results += 1
                solved_constraints += 1
            elif result_status == 'unsat':
                unsat_results += 1
                solved_constraints += 1
            elif result_status == 'unknown':
                unknown_results += 1
                unsolved_constraints += 1
            else:
                error_results += 1
                unsolved_constraints += 1
            
            # 统计方法
            method_stats[method_status] += 1
            
            # 统计状态
            status_stats[result_status] += 1
            
            # 时间分布统计
            if isinstance(solve_time, (int, float)) and solve_time > 0:
                all_solve_times.append(solve_time)
                
                # 时间分段统计
                if solve_time <= 10:
                    time_distribution['0-10s'] += 1
                elif solve_time <= 30:
                    time_distribution['10-30s'] += 1
                elif solve_time <= 60:
                    time_distribution['30-60s'] += 1
                elif solve_time <= 120:
                    time_distribution['60-120s'] += 1
                elif solve_time <= 300:
                    time_distribution['120-300s'] += 1
                elif solve_time <= 600:
                    time_distribution['300-600s'] += 1
                elif solve_time <= 1200:
                    time_distribution['600-1200s'] += 1
                else:
                    time_distribution['>1200s'] += 1
    
    # 打印总体统计
    print("Overall Statistics:")
    print(f"  Total constraints: {total_constraints}")
    print(f"  Solved constraints: {solved_constraints} ({solved_constraints/total_constraints*100:.1f}%)")
    print(f"  Unsolved constraints: {unsolved_constraints} ({unsolved_constraints/total_constraints*100:.1f}%)")
    print()
    
    print("Result Distribution:")
    print(f"  SAT: {sat_results} ({sat_results/total_constraints*100:.1f}%)")
    print(f"  UNSAT: {unsat_results} ({unsat_results/total_constraints*100:.1f}%)")
    print(f"  UNKNOWN: {unknown_results} ({unknown_results/total_constraints*100:.1f}%)")
    print(f"  ERROR: {error_results} ({error_results/total_constraints*100:.1f}%)")
    print()
    
    print("Method Distribution:")
    for method, count in method_stats.items():
        print(f"  {method}: {count} ({count/total_constraints*100:.1f}%)")
    print()
    
    # 时间分布统计
    if all_solve_times:
        print("Time Distribution (for solved constraints):")
        sorted_time_ranges = [
            '0-10s', '10-30s', '30-60s', '60-120s', 
            '120-300s', '300-600s', '600-1200s', '>1200s'
        ]
        
        for time_range in sorted_time_ranges:
            count = time_distribution[time_range]
            if count > 0:
                print(f"  {time_range}: {count} constraints")
        
        print()
        print("Time Statistics:")
        print(f"  Mean solve time: {np.mean(all_solve_times):.2f}s")
        print(f"  Median solve time: {np.median(all_solve_times):.2f}s")
        print(f"  Min solve time: {np.min(all_solve_times):.2f}s")
        print(f"  Max solve time: {np.max(all_solve_times):.2f}s")
        print(f"  Std solve time: {np.std(all_solve_times):.2f}s")
    
    # 成功率统计
    if solved_constraints > 0:
        success_rate = solved_constraints / total_constraints * 100
        print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    return {
        'total': total_constraints,
        'solved': solved_constraints,
        'unsolved': unsolved_constraints,
        'sat': sat_results,
        'unsat': unsat_results,
        'unknown': unknown_results,
        'error': error_results,
        'method_stats': dict(method_stats),
        'time_distribution': dict(time_distribution),
        'time_stats': {
            'mean': np.mean(all_solve_times) if all_solve_times else 0,
            'median': np.median(all_solve_times) if all_solve_times else 0,
            'min': np.min(all_solve_times) if all_solve_times else 0,
            'max': np.max(all_solve_times) if all_solve_times else 0,
            'std': np.std(all_solve_times) if all_solve_times else 0
        }
    }

def analyze_by_method(data):
    """按方法分析结果"""
    print("\n" + "=" * 80)
    print("Analysis by Method")
    print("=" * 80)
    
    method_data = defaultdict(lambda: {
        'total': 0, 'solved': 0, 'sat': 0, 'unsat': 0, 
        'unknown': 0, 'error': 0, 'times': []
    })
    
    for constraint_path, result in data.items():
        if isinstance(result, list) and len(result) >= 8:
            method = result[7] if len(result) > 7 else 'unknown'
            result_status = result[0] if len(result) > 0 else 'unknown'
            solve_time = result[1] if len(result) > 1 else 0
            
            method_data[method]['total'] += 1
            
            if result_status == 'sat':
                method_data[method]['sat'] += 1
                method_data[method]['solved'] += 1
            elif result_status == 'unsat':
                method_data[method]['unsat'] += 1
                method_data[method]['solved'] += 1
            elif result_status == 'unknown':
                method_data[method]['unknown'] += 1
            else:
                method_data[method]['error'] += 1
            
            if isinstance(solve_time, (int, float)) and solve_time > 0:
                method_data[method]['times'].append(solve_time)
    
    for method, stats in method_data.items():
        if stats['total'] > 0:
            print(f"\nMethod: {method}")
            print(f"  Total: {stats['total']}")
            print(f"  Solved: {stats['solved']} ({stats['solved']/stats['total']*100:.1f}%)")
            print(f"  SAT: {stats['sat']} ({stats['sat']/stats['total']*100:.1f}%)")
            print(f"  UNSAT: {stats['unsat']} ({stats['unsat']/stats['total']*100:.1f}%)")
            print(f"  UNKNOWN: {stats['unknown']} ({stats['unknown']/stats['total']*100:.1f}%)")
            print(f"  ERROR: {stats['error']} ({stats['error']/stats['total']*100:.1f}%)")
            
            if stats['times']:
                print(f"  Avg Time: {np.mean(stats['times']):.2f}s")
                print(f"  Median Time: {np.median(stats['times']):.2f}s")

def main():
    """主函数"""
    file_path = "advanced_solver_results_all.json"
    
    if not os.path.exists(file_path):
        print(f"File {file_path} not found!")
        return
    
    print(f"Analyzing file: {file_path}")
    
    # 加载数据
    data = load_json_file(file_path)
    if data is None:
        return
    
    # 分析结果
    stats = analyze_solver_results(data)
    
    # 按方法分析
    analyze_by_method(data)
    
    # 保存统计结果
    output_file = "advanced_results_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f"\nAnalysis results saved to: {output_file}")

if __name__ == "__main__":
    main() 