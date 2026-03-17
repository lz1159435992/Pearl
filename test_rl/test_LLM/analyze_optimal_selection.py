"""
计算"最优选择策略"的平均百分比
即：如果每次都选择出现百分比最高的变量，平均百分比是多少？

对比三种策略：
1. 最优策略（总是选最高百分比）
2. LLM实际选择
3. 随机选择（期望值）
"""
import json
import os
import numpy as np
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')


def load_results():
    """加载实验结果"""
    results_file = os.path.join(OUTPUT_DIR, 'experiment_results.json')
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_optimal_strategy(results):
    """
    分析三种策略的平均百分比
    """
    # 存储各策略的百分比
    optimal_percentages_first = []  # 最优策略（第一次选择）
    optimal_percentages_second = []  # 最优策略（第二次选择）
    llm_percentages_first = []  # LLM实际选择（第一次）
    llm_percentages_second = []  # LLM实际选择（第二次）
    random_percentages = []  # 随机基线
    
    # 按文件分组，避免重复计算随机基线
    files_processed = set()
    
    for result in results:
        # 获取所有变量的出现次数
        all_var_counts = result.get('all_original_var_counts', {})
        if not all_var_counts:
            continue
        
        total_count = sum(all_var_counts.values())
        if total_count == 0:
            continue
        
        # 计算每个变量的百分比
        var_percentages = {var: (count / total_count) * 100 
                          for var, count in all_var_counts.items()}
        
        # 按百分比降序排序
        sorted_vars = sorted(var_percentages.items(), key=lambda x: x[1], reverse=True)
        
        # 最优策略：第一次选择最高百分比的变量
        if sorted_vars:
            optimal_first_pct = sorted_vars[0][1]
            optimal_percentages_first.append(optimal_first_pct)
            
            # 最优策略：第二次选择第二高百分比的变量
            if len(sorted_vars) > 1:
                optimal_second_pct = sorted_vars[1][1]
                optimal_percentages_second.append(optimal_second_pct)
        
        # LLM实际选择（第一次）
        first_selected_var = result.get('first_selected_original_var')
        if first_selected_var and first_selected_var in var_percentages:
            llm_first_pct = var_percentages[first_selected_var]
            llm_percentages_first.append(llm_first_pct)
        
        # LLM实际选择（第二次）
        second_selected_var = result.get('second_selected_original_var')
        if second_selected_var and second_selected_var in var_percentages:
            llm_second_pct = var_percentages[second_selected_var]
            llm_percentages_second.append(llm_second_pct)
        
        # 随机基线：每个文件只计算一次
        file_idx = result.get('file_index')
        if file_idx not in files_processed:
            files_processed.add(file_idx)
            for var, pct in var_percentages.items():
                random_percentages.append(pct)
    
    return {
        'optimal_first': optimal_percentages_first,
        'optimal_second': optimal_percentages_second,
        'llm_first': llm_percentages_first,
        'llm_second': llm_percentages_second,
        'random': random_percentages
    }


def print_report(data):
    """打印分析报告"""
    print("=" * 80)
    print("三种选择策略的平均出现百分比对比")
    print("=" * 80)
    print()
    
    # 计算统计量
    stats = {}
    for key, values in data.items():
        if values:
            stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }
    
    # 打印第一次选择对比
    print("## 第一次选择")
    print("-" * 60)
    print(f"{'策略':<20} {'平均百分比':<15} {'标准差':<15} {'样本数':<10}")
    print("-" * 60)
    
    if 'optimal_first' in stats:
        s = stats['optimal_first']
        print(f"{'最优策略（选最高）':<20} {s['mean']:.4f}%{'':<8} {s['std']:.4f}%{'':<8} {s['count']}")
    
    if 'llm_first' in stats:
        s = stats['llm_first']
        print(f"{'LLM实际选择':<20} {s['mean']:.4f}%{'':<8} {s['std']:.4f}%{'':<8} {s['count']}")
    
    if 'random' in stats:
        s = stats['random']
        print(f"{'随机选择（期望）':<20} {s['mean']:.4f}%{'':<8} {s['std']:.4f}%{'':<8} {s['count']}")
    
    print()
    
    # 计算LLM相对于最优的效率
    if 'optimal_first' in stats and 'llm_first' in stats and 'random' in stats:
        optimal_mean = stats['optimal_first']['mean']
        llm_mean = stats['llm_first']['mean']
        random_mean = stats['random']['mean']
        
        # LLM达到最优的百分比
        llm_efficiency = (llm_mean - random_mean) / (optimal_mean - random_mean) * 100
        
        print(f"LLM选择效率（相对于最优）: {llm_efficiency:.2f}%")
        print(f"  - 最优策略比随机高: {optimal_mean - random_mean:.4f}%")
        print(f"  - LLM比随机高: {llm_mean - random_mean:.4f}%")
        print(f"  - LLM达到了最优提升的 {llm_efficiency:.2f}%")
    
    print()
    
    # 打印第二次选择对比
    print("## 第二次选择")
    print("-" * 60)
    print(f"{'策略':<20} {'平均百分比':<15} {'标准差':<15} {'样本数':<10}")
    print("-" * 60)
    
    if 'optimal_second' in stats:
        s = stats['optimal_second']
        print(f"{'最优策略（选第二高）':<20} {s['mean']:.4f}%{'':<8} {s['std']:.4f}%{'':<8} {s['count']}")
    
    if 'llm_second' in stats:
        s = stats['llm_second']
        print(f"{'LLM实际选择':<20} {s['mean']:.4f}%{'':<8} {s['std']:.4f}%{'':<8} {s['count']}")
    
    if 'random' in stats:
        s = stats['random']
        print(f"{'随机选择（期望）':<20} {s['mean']:.4f}%{'':<8} {s['std']:.4f}%{'':<8} {s['count']}")
    
    print()
    
    # 打印汇总表格
    print("=" * 80)
    print("## 汇总对比")
    print("=" * 80)
    print()
    print(f"{'选择次序':<10} {'最优策略':<15} {'LLM选择':<15} {'随机期望':<15} {'LLM/最优':<10}")
    print("-" * 65)
    
    if all(k in stats for k in ['optimal_first', 'llm_first', 'random']):
        opt = stats['optimal_first']['mean']
        llm = stats['llm_first']['mean']
        rnd = stats['random']['mean']
        ratio = llm / opt * 100
        print(f"{'第一次':<10} {opt:.4f}%{'':<8} {llm:.4f}%{'':<8} {rnd:.4f}%{'':<8} {ratio:.1f}%")
    
    if all(k in stats for k in ['optimal_second', 'llm_second', 'random']):
        opt = stats['optimal_second']['mean']
        llm = stats['llm_second']['mean']
        rnd = stats['random']['mean']
        ratio = llm / opt * 100
        print(f"{'第二次':<10} {opt:.4f}%{'':<8} {llm:.4f}%{'':<8} {rnd:.4f}%{'':<8} {ratio:.1f}%")
    
    print()
    print("=" * 80)
    print("## 关键结论")
    print("=" * 80)
    print()
    
    if 'optimal_first' in stats and 'llm_first' in stats and 'random' in stats:
        opt = stats['optimal_first']['mean']
        llm = stats['llm_first']['mean']
        rnd = stats['random']['mean']
        
        print(f"1. 最优策略（总是选最高百分比）的平均百分比: {opt:.4f}%")
        print(f"2. LLM实际选择的平均百分比: {llm:.4f}%")
        print(f"3. 随机选择的期望百分比: {rnd:.4f}%")
        print()
        print(f"4. 最优策略是随机的 {opt/rnd:.2f} 倍")
        print(f"5. LLM选择是随机的 {llm/rnd:.2f} 倍")
        print(f"6. LLM选择达到最优的 {llm/opt*100:.1f}%")
    
    return stats


def main():
    print("加载实验结果...")
    results = load_results()
    print(f"共加载 {len(results)} 条记录")
    print()
    
    print("分析三种策略...")
    data = analyze_optimal_strategy(results)
    
    print()
    stats = print_report(data)
    
    return stats


if __name__ == '__main__':
    main()
