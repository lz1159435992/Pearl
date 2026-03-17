"""
分析第二次选择与最优策略（选择次高频变量）的对比
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


def analyze_second_vs_optimal(results):
    """
    分析第二次选择与最优策略的对比
    最优策略：在第一次选择后，选择剩余变量中频次最高的（即次高频变量）
    """
    
    # 统计数据
    total_valid = 0
    
    # 第二次选择是否选中了次高频变量
    selected_second_highest = 0  # 选中了次高频（排除第一次选择后的最高频）
    
    # 第二次选择的排名分布（排除第一次选择后的排名）
    second_ranks = []
    
    # 百分比对比
    llm_second_pcts = []
    optimal_second_pcts = []  # 最优策略：选择次高频
    random_pcts = []
    
    for result in results:
        all_var_counts = result.get('all_original_var_counts', {})
        if not all_var_counts:
            continue
        
        total_count = sum(all_var_counts.values())
        if total_count == 0:
            continue
        
        first_var = result.get('first_selected_original_var')
        second_var = result.get('second_selected_original_var')
        
        if not first_var or not second_var:
            continue
        
        if first_var not in all_var_counts or second_var not in all_var_counts:
            continue
        
        total_valid += 1
        
        # 按频次降序排序所有变量
        sorted_vars = sorted(all_var_counts.items(), key=lambda x: x[1], reverse=True)
        var_to_rank = {var: rank + 1 for rank, (var, _) in enumerate(sorted_vars)}
        
        # 排除第一次选择后的剩余变量
        remaining_vars = [(var, count) for var, count in sorted_vars if var != first_var]
        
        if not remaining_vars:
            continue
        
        # 最优策略：选择剩余变量中频次最高的
        optimal_second_var = remaining_vars[0][0]
        optimal_second_count = remaining_vars[0][1]
        optimal_second_pct = (optimal_second_count / total_count) * 100
        optimal_second_pcts.append(optimal_second_pct)
        
        # LLM第二次选择的百分比
        llm_second_count = all_var_counts[second_var]
        llm_second_pct = (llm_second_count / total_count) * 100
        llm_second_pcts.append(llm_second_pct)
        
        # 检查LLM是否选中了最优（次高频）
        if second_var == optimal_second_var:
            selected_second_highest += 1
        
        # 计算第二次选择在剩余变量中的排名
        remaining_var_to_rank = {var: rank + 1 for rank, (var, _) in enumerate(remaining_vars)}
        if second_var in remaining_var_to_rank:
            second_ranks.append(remaining_var_to_rank[second_var])
        
        # 随机基线（剩余变量的平均百分比）
        for var, count in remaining_vars:
            random_pcts.append((count / total_count) * 100)
    
    return {
        'total_valid': total_valid,
        'selected_second_highest': selected_second_highest,
        'second_ranks': second_ranks,
        'llm_second_pcts': llm_second_pcts,
        'optimal_second_pcts': optimal_second_pcts,
        'random_pcts': random_pcts
    }


def print_report(data):
    """打印分析报告"""
    print("=" * 80)
    print("第二次选择 vs 最优策略（选择次高频变量）分析")
    print("=" * 80)
    print()
    
    total = data['total_valid']
    selected_optimal = data['selected_second_highest']
    
    print("## 1. 基本统计")
    print("-" * 60)
    print(f"有效记录数: {total}")
    print(f"LLM选中次高频变量（最优）: {selected_optimal} ({selected_optimal/total*100:.2f}%)")
    print(f"LLM未选中次高频变量: {total - selected_optimal} ({(total-selected_optimal)/total*100:.2f}%)")
    print()
    
    print("## 2. 百分比对比")
    print("-" * 60)
    if data['llm_second_pcts'] and data['optimal_second_pcts']:
        llm_mean = np.mean(data['llm_second_pcts'])
        optimal_mean = np.mean(data['optimal_second_pcts'])
        random_mean = np.mean(data['random_pcts'])
        
        print(f"{'策略':<25} {'平均百分比':<15} {'与随机的倍数':<15}")
        print("-" * 55)
        print(f"{'最优策略（选次高频）':<25} {optimal_mean:.4f}%{'':<8} {optimal_mean/random_mean:.2f}x")
        print(f"{'LLM第二次选择':<25} {llm_mean:.4f}%{'':<8} {llm_mean/random_mean:.2f}x")
        print(f"{'随机选择（剩余变量）':<25} {random_mean:.4f}%{'':<8} 1.00x")
        print()
        
        # LLM达到最优的效率
        efficiency = (llm_mean - random_mean) / (optimal_mean - random_mean) * 100 if optimal_mean != random_mean else 0
        print(f"LLM达到最优的效率: {efficiency:.1f}%")
        print(f"LLM/最优: {llm_mean/optimal_mean*100:.1f}%")
    print()
    
    print("## 3. 排名分析（在剩余变量中的排名）")
    print("-" * 60)
    if data['second_ranks']:
        ranks = np.array(data['second_ranks'])
        print(f"平均排名: {np.mean(ranks):.2f}")
        print(f"中位数排名: {np.median(ranks):.2f}")
        print(f"排名标准差: {np.std(ranks):.2f}")
        print()
        
        # 排名分布
        print("排名分布（1=剩余变量中最高频）:")
        for rank in range(1, min(11, int(max(ranks)) + 1)):
            count = sum(1 for r in ranks if r == rank)
            pct = count / len(ranks) * 100
            bar = '█' * int(pct / 2)
            print(f"  排名{rank:2d}: {count:4d} ({pct:5.1f}%) {bar}")
        
        # 前N名的累计比例
        print()
        print("累计比例:")
        for n in [1, 3, 5, 10]:
            count = sum(1 for r in ranks if r <= n)
            print(f"  前{n}名: {count} ({count/len(ranks)*100:.1f}%)")
    print()
    
    print("=" * 80)
    print("## 结论")
    print("=" * 80)
    print()
    
    if total > 0:
        optimal_rate = selected_optimal / total * 100
        print(f"1. LLM第二次选择次高频变量（最优）的比例: {optimal_rate:.2f}%")
        
        if data['llm_second_pcts'] and data['optimal_second_pcts']:
            llm_mean = np.mean(data['llm_second_pcts'])
            optimal_mean = np.mean(data['optimal_second_pcts'])
            print(f"2. LLM第二次选择的平均百分比: {llm_mean:.4f}%")
            print(f"3. 最优策略的平均百分比: {optimal_mean:.4f}%")
            print(f"4. LLM达到最优的: {llm_mean/optimal_mean*100:.1f}%")
        
        if data['second_ranks']:
            ranks = np.array(data['second_ranks'])
            print(f"5. LLM第二次选择的平均排名: {np.mean(ranks):.2f}")


def main():
    print("加载实验结果...")
    results = load_results()
    print(f"共加载 {len(results)} 条记录")
    print()
    
    print("分析第二次选择与最优策略...")
    data = analyze_second_vs_optimal(results)
    
    print()
    print_report(data)


if __name__ == '__main__':
    main()
