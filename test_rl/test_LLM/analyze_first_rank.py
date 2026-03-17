"""
分析LLM第一次选择变量的排名分布
"""
import json
import os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')


def load_results():
    """加载实验结果"""
    results_file = os.path.join(OUTPUT_DIR, 'experiment_results.json')
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_first_rank(results):
    """分析第一次选择的排名分布"""
    
    first_ranks = []
    total_vars_list = []
    
    for result in results:
        all_var_counts = result.get('all_original_var_counts', {})
        if not all_var_counts:
            continue
        
        first_var = result.get('first_selected_original_var')
        if not first_var or first_var not in all_var_counts:
            continue
        
        # 按频次降序排序
        sorted_vars = sorted(all_var_counts.items(), key=lambda x: x[1], reverse=True)
        var_to_rank = {var: rank + 1 for rank, (var, _) in enumerate(sorted_vars)}
        
        first_ranks.append(var_to_rank[first_var])
        total_vars_list.append(len(all_var_counts))
    
    return first_ranks, total_vars_list


def print_report(first_ranks, total_vars_list):
    """打印分析报告"""
    print("=" * 80)
    print("LLM第一次选择变量的排名分布")
    print("=" * 80)
    print()
    
    ranks = np.array(first_ranks)
    total_vars = np.array(total_vars_list)
    
    print("## 1. 基本统计")
    print("-" * 60)
    print(f"有效记录数: {len(ranks)}")
    print(f"平均排名: {np.mean(ranks):.2f}")
    print(f"中位数排名: {np.median(ranks):.2f}")
    print(f"排名标准差: {np.std(ranks):.2f}")
    print(f"最小排名: {np.min(ranks)}")
    print(f"最大排名: {np.max(ranks)}")
    print()
    print(f"平均变量数: {np.mean(total_vars):.1f}")
    print()
    
    print("## 2. 排名分布（1=最高频）")
    print("-" * 60)
    max_show = min(20, int(max(ranks)))
    for rank in range(1, max_show + 1):
        count = sum(1 for r in ranks if r == rank)
        pct = count / len(ranks) * 100
        bar = '█' * int(pct / 2)
        print(f"  排名{rank:2d}: {count:4d} ({pct:5.2f}%) {bar}")
    
    # 显示更高排名的汇总
    higher_count = sum(1 for r in ranks if r > max_show)
    if higher_count > 0:
        print(f"  排名>{max_show}: {higher_count:4d} ({higher_count/len(ranks)*100:5.2f}%)")
    print()
    
    print("## 3. 累计比例")
    print("-" * 60)
    for n in [1, 3, 5, 10, 20, 50, 100]:
        count = sum(1 for r in ranks if r <= n)
        print(f"  前{n:3d}名: {count:4d} ({count/len(ranks)*100:5.1f}%)")
    print()
    
    print("## 4. 与随机选择对比")
    print("-" * 60)
    # 随机选择的期望排名 = (变量数+1)/2
    expected_random_rank = np.mean((total_vars + 1) / 2)
    print(f"LLM平均排名: {np.mean(ranks):.2f}")
    print(f"随机选择期望排名: {expected_random_rank:.2f}")
    print(f"LLM比随机好: {expected_random_rank - np.mean(ranks):.2f} 名")
    print()
    
    # 选中最高频的比例
    top1_rate = sum(1 for r in ranks if r == 1) / len(ranks) * 100
    random_top1_rate = 100 / np.mean(total_vars)
    print(f"LLM选中最高频的比例: {top1_rate:.2f}%")
    print(f"随机选中最高频的期望: {random_top1_rate:.2f}%")
    print(f"LLM/随机: {top1_rate/random_top1_rate:.1f}x")
    print()
    
    print("=" * 80)
    print("## 结论")
    print("=" * 80)
    print()
    print(f"1. LLM第一次选择的平均排名: {np.mean(ranks):.2f}")
    print(f"2. LLM选中最高频变量的比例: {top1_rate:.2f}%")
    print(f"3. LLM选中前3名的比例: {sum(1 for r in ranks if r <= 3)/len(ranks)*100:.2f}%")
    print(f"4. LLM选中前10名的比例: {sum(1 for r in ranks if r <= 10)/len(ranks)*100:.2f}%")


def main():
    print("加载实验结果...")
    results = load_results()
    print(f"共加载 {len(results)} 条记录")
    print()
    
    first_ranks, total_vars = analyze_first_rank(results)
    print_report(first_ranks, total_vars)


if __name__ == '__main__':
    main()
