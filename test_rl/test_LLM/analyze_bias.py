"""
分析LLM选择变量是否偏向高百分比变量（基于相对频次）
对比实际选择与随机选择的差异
注意：使用变量出现次数占所有变量出现次数总和的百分比进行排名，标准化了不同大小文件的影响
"""
import json
import numpy as np
from scipy import stats
import os

def analyze_bias():
    """分析LLM选择变量是否偏向高百分比变量"""
    
    # 加载结果
    results_file = 'output/experiment_results.json'
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print("="*80)
    print("LLM变量选择偏向性分析（基于相对频次百分比）")
    print("="*80)
    print("注意：排名基于变量出现次数占所有变量出现次数总和的百分比")
    print("     标准化了不同大小约束文件的影响，更加公平合理")
    print()
    
    # 收集数据
    first_ranks = []
    second_ranks = []
    total_vars_per_file = []
    
    # 按文件分组，计算每个文件的期望排名
    file_var_counts = {}  # {file_index: [var_count_list]}
    
    for result in results:
        all_counts_dict = result.get('all_original_var_counts', {})
        if not all_counts_dict:
            continue
        
        file_idx = result.get('file_index')
        total_vars = len(all_counts_dict)
        
        if file_idx not in file_var_counts:
            file_var_counts[file_idx] = []
        file_var_counts[file_idx].append(total_vars)
        
        total_vars_per_file.append(total_vars)
        
        # 计算所有变量的总出现次数
        total_count = sum(all_counts_dict.values())
        if total_count == 0:
            continue
        
        # 将绝对频次转换为百分比（相对频次）
        all_percentages_dict = {var: count / total_count * 100 
                                for var, count in all_counts_dict.items()}
        all_percentages = sorted(all_percentages_dict.values(), reverse=True)
        
        if len(all_percentages) == 0:
            continue
        
        # 获取被选择变量的百分比并计算排名
        first_count = result.get('first_selected_original_var_count')
        if first_count is not None:
            first_percentage = first_count / total_count * 100
            rank = sum(1 for p in all_percentages if p > first_percentage) + 1
            first_ranks.append(rank)
        
        second_count = result.get('second_selected_original_var_count')
        if second_count is not None:
            second_percentage = second_count / total_count * 100
            rank = sum(1 for p in all_percentages if p > second_percentage) + 1
            second_ranks.append(rank)
    
    print(f"总文件数: {len(file_var_counts)}")
    print(f"总实验次数: {len(results)}")
    print(f"平均变量数: {np.mean(total_vars_per_file):.2f}")
    print(f"变量数范围: {np.min(total_vars_per_file)} - {np.max(total_vars_per_file)}")
    print()
    
    print("="*80)
    print("1. 随机选择的期望表现")
    print("="*80)
    avg_vars = np.mean(total_vars_per_file)
    expected_mean_rank = (avg_vars + 1) / 2
    print(f"平均变量数: {avg_vars:.1f}")
    print(f"如果完全随机选择，期望平均排名: {expected_mean_rank:.1f}")
    print(f"期望中位数排名: {avg_vars / 2:.1f}")
    
    # 计算随机选择前N个的概率
    print(f"\n如果完全随机选择，选择各排名区间的期望概率:")
    for n in [1, 3, 10, 20]:
        expected_prob = n / avg_vars * 100
        print(f"  排名 <= {n}: 期望概率 {expected_prob:.2f}%")
    print()
    
    print("="*80)
    print("2. 实际观察到的选择行为")
    print("="*80)
    
    if first_ranks:
        print("第一次选择:")
        actual_mean = np.mean(first_ranks)
        actual_median = np.median(first_ranks)
        print(f"  实际平均排名: {actual_mean:.2f} (期望: {expected_mean_rank:.2f})")
        print(f"  实际中位数排名: {actual_median:.1f} (期望: {avg_vars/2:.1f})")
        print(f"  排名差异: {actual_mean - expected_mean_rank:.2f} ({((actual_mean - expected_mean_rank) / expected_mean_rank * 100):.1f}%)")
        
        top1_ratio = sum(1 for r in first_ranks if r == 1) / len(first_ranks) * 100
        top3_ratio = sum(1 for r in first_ranks if r <= 3) / len(first_ranks) * 100
        top10_ratio = sum(1 for r in first_ranks if r <= 10) / len(first_ranks) * 100
        top20_ratio = sum(1 for r in first_ranks if r <= 20) / len(first_ranks) * 100
        
        print(f"\n  实际选择各排名区间的比例:")
        print(f"    排名 = 1 (最高百分比): {top1_ratio:.2f}% (期望: {1/avg_vars*100:.2f}%)")
        print(f"    排名 <= 3: {top3_ratio:.2f}% (期望: {3/avg_vars*100:.2f}%)")
        print(f"    排名 <= 10: {top10_ratio:.2f}% (期望: {10/avg_vars*100:.2f}%)")
        print(f"    排名 <= 20: {top20_ratio:.2f}% (期望: {20/avg_vars*100:.2f}%)")
        print()
    
    if second_ranks:
        print("第二次选择:")
        actual_mean = np.mean(second_ranks)
        actual_median = np.median(second_ranks)
        print(f"  实际平均排名: {actual_mean:.2f} (期望: {expected_mean_rank:.2f})")
        print(f"  实际中位数排名: {actual_median:.1f} (期望: {avg_vars/2:.1f})")
        print(f"  排名差异: {actual_mean - expected_mean_rank:.2f} ({((actual_mean - expected_mean_rank) / expected_mean_rank * 100):.1f}%)")
        
        top1_ratio = sum(1 for r in second_ranks if r == 1) / len(second_ranks) * 100
        top3_ratio = sum(1 for r in second_ranks if r <= 3) / len(second_ranks) * 100
        top10_ratio = sum(1 for r in second_ranks if r <= 10) / len(second_ranks) * 100
        top20_ratio = sum(1 for r in second_ranks if r <= 20) / len(second_ranks) * 100
        
        print(f"\n  实际选择各排名区间的比例:")
        print(f"    排名 = 1 (最高百分比): {top1_ratio:.2f}% (期望: {1/avg_vars*100:.2f}%)")
        print(f"    排名 <= 3: {top3_ratio:.2f}% (期望: {3/avg_vars*100:.2f}%)")
        print(f"    排名 <= 10: {top10_ratio:.2f}% (期望: {10/avg_vars*100:.2f}%)")
        print(f"    排名 <= 20: {top20_ratio:.2f}% (期望: {20/avg_vars*100:.2f}%)")
        print()
    
    print("="*80)
    print("3. 统计显著性检验")
    print("="*80)
    print("H0 (零假设): LLM的选择是随机的（平均排名 = 期望排名）")
    print("H1 (备择假设): LLM的选择偏向高百分比变量（平均排名 < 期望排名）")
    print()
    
    if first_ranks and len(first_ranks) > 1:
        expected_rank = expected_mean_rank
        actual_mean_rank = np.mean(first_ranks)
        
        # 单样本t检验（单侧）
        t_stat, p_value = stats.ttest_1samp(first_ranks, expected_rank, alternative='less')
        
        print("第一次选择:")
        print(f"  实际平均排名: {actual_mean_rank:.2f}")
        print(f"  期望平均排名（随机）: {expected_rank:.2f}")
        print(f"  t统计量: {t_stat:.4f}")
        print(f"  p值: {p_value:.6f}")
        if p_value < 0.001:
            print(f"  ✓✓✓ 极显著偏向高百分比变量 (p < 0.001)")
        elif p_value < 0.01:
            print(f"  ✓✓ 非常显著偏向高百分比变量 (p < 0.01)")
        elif p_value < 0.05:
            print(f"  ✓ 显著偏向高百分比变量 (p < 0.05)")
        else:
            print(f"  ✗ 未显著偏离随机选择 (p >= 0.05)")
        
        # 计算效应量（Cohen's d）
        cohens_d = (expected_rank - actual_mean_rank) / np.std(first_ranks)
        print(f"  效应量 (Cohen's d): {cohens_d:.4f}")
        if abs(cohens_d) < 0.2:
            effect_size = "小"
        elif abs(cohens_d) < 0.5:
            effect_size = "中等"
        elif abs(cohens_d) < 0.8:
            effect_size = "大"
        else:
            effect_size = "很大"
        print(f"  效应量大小: {effect_size}")
        print()
    
    if second_ranks and len(second_ranks) > 1:
        expected_rank = expected_mean_rank
        actual_mean_rank = np.mean(second_ranks)
        
        t_stat, p_value = stats.ttest_1samp(second_ranks, expected_rank, alternative='less')
        
        print("第二次选择:")
        print(f"  实际平均排名: {actual_mean_rank:.2f}")
        print(f"  期望平均排名（随机）: {expected_rank:.2f}")
        print(f"  t统计量: {t_stat:.4f}")
        print(f"  p值: {p_value:.6f}")
        if p_value < 0.001:
            print(f"  ✓✓✓ 极显著偏向高百分比变量 (p < 0.001)")
        elif p_value < 0.01:
            print(f"  ✓✓ 非常显著偏向高百分比变量 (p < 0.01)")
        elif p_value < 0.05:
            print(f"  ✓ 显著偏向高百分比变量 (p < 0.05)")
        else:
            print(f"  ✗ 未显著偏离随机选择 (p >= 0.05)")
        
        cohens_d = (expected_rank - actual_mean_rank) / np.std(second_ranks)
        print(f"  效应量 (Cohen's d): {cohens_d:.4f}")
        if abs(cohens_d) < 0.2:
            effect_size = "小"
        elif abs(cohens_d) < 0.5:
            effect_size = "中等"
        elif abs(cohens_d) < 0.8:
            effect_size = "大"
        else:
            effect_size = "很大"
        print(f"  效应量大小: {effect_size}")
        print()
    
    print("="*80)
    print("4. 结论总结")
    print("="*80)
    
    if first_ranks:
        actual_mean = np.mean(first_ranks)
        expected_mean = expected_mean_rank
        reduction = (expected_mean - actual_mean) / expected_mean * 100
        top10_ratio = sum(1 for r in first_ranks if r <= 10) / len(first_ranks) * 100
        expected_top10 = 10 / avg_vars * 100
        
        print("\n第一次选择:")
        if actual_mean < expected_mean:
            print(f"  ✓ LLM的选择偏向高百分比变量")
            print(f"    - 平均排名降低了 {reduction:.1f}%")
            print(f"    - 选择前10高百分比变量的比例 ({top10_ratio:.1f}%) 是随机期望 ({expected_top10:.1f}%) 的 {top10_ratio/expected_top10:.1f}倍")
        else:
            print(f"  ✗ LLM的选择未明显偏向高百分比变量")
        print()
    
    if second_ranks:
        actual_mean = np.mean(second_ranks)
        expected_mean = expected_mean_rank
        reduction = (expected_mean - actual_mean) / expected_mean * 100
        top10_ratio = sum(1 for r in second_ranks if r <= 10) / len(second_ranks) * 100
        expected_top10 = 10 / avg_vars * 100
        
        print("第二次选择:")
        if actual_mean < expected_mean:
            print(f"  ✓ LLM的选择偏向高百分比变量")
            print(f"    - 平均排名降低了 {reduction:.1f}%")
            print(f"    - 选择前10高百分比变量的比例 ({top10_ratio:.1f}%) 是随机期望 ({expected_top10:.1f}%) 的 {top10_ratio/expected_top10:.1f}倍")
        else:
            print(f"  ✗ LLM的选择未明显偏向高百分比变量")

if __name__ == "__main__":
    analyze_bias()


