"""
LLM变量选择偏向性分析脚本 - 使用百分比统计
证明LLM选择变量时偏向于出现次数百分比较高的变量

分析方法：
1. 将变量出现次数转换为百分比（该变量出现次数 / 所有变量总出现次数）
2. 与随机选择基线对比
3. 使用统计检验（t检验、Mann-Whitney U检验）证明显著性
4. 计算效应量（Cohen's d）
5. 可视化分析
"""
import json
import os
import sys
import numpy as np
from collections import defaultdict
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')


def load_results():
    """加载实验结果"""
    results_file = os.path.join(OUTPUT_DIR, 'experiment_results.json')
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_percentage(var_count, all_var_counts):
    """
    计算变量出现次数的百分比
    
    参数:
    - var_count: 该变量的出现次数
    - all_var_counts: 所有变量的出现次数字典 {var: count}
    
    返回:
    - 百分比 (0-100)
    """
    total = sum(all_var_counts.values())
    if total == 0:
        return 0
    return (var_count / total) * 100


def analyze_with_percentage(results):
    """
    使用百分比分析LLM选择偏向性
    
    返回:
    - selected_percentages: LLM选择的变量的出现百分比列表
    - random_percentages: 随机选择的期望百分比列表（每个变量的百分比）
    - all_percentages_per_file: 每个文件中所有变量的百分比（用于计算随机基线）
    """
    selected_percentages_first = []  # 第一次选择
    selected_percentages_second = []  # 第二次选择
    random_baseline_percentages = []  # 随机基线（所有变量的百分比）
    
    # 按文件分组，避免重复计算同一文件的随机基线
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
        
        # 第一次选择
        first_selected_var = result.get('first_selected_original_var')
        if first_selected_var and first_selected_var in all_var_counts:
            first_count = all_var_counts[first_selected_var]
            first_pct = (first_count / total_count) * 100
            selected_percentages_first.append(first_pct)
        
        # 第二次选择
        second_selected_var = result.get('second_selected_original_var')
        if second_selected_var and second_selected_var in all_var_counts:
            second_count = all_var_counts[second_selected_var]
            second_pct = (second_count / total_count) * 100
            selected_percentages_second.append(second_pct)
        
        # 随机基线：每个文件只计算一次
        file_idx = result.get('file_index')
        if file_idx not in files_processed:
            files_processed.add(file_idx)
            # 添加该文件中所有变量的百分比作为随机基线
            for var, pct in var_percentages.items():
                random_baseline_percentages.append(pct)
    
    return (selected_percentages_first, selected_percentages_second, 
            random_baseline_percentages)


def statistical_tests(selected_pcts, random_pcts, label=""):
    """
    执行统计检验
    
    参数:
    - selected_pcts: LLM选择的变量百分比
    - random_pcts: 随机基线百分比
    - label: 标签（第一次/第二次选择）
    
    返回:
    - 统计结果字典
    """
    selected = np.array(selected_pcts)
    random = np.array(random_pcts)
    
    results = {
        'label': label,
        'n_selected': len(selected),
        'n_random': len(random),
        'mean_selected': np.mean(selected),
        'mean_random': np.mean(random),
        'std_selected': np.std(selected),
        'std_random': np.std(random),
        'median_selected': np.median(selected),
        'median_random': np.median(random),
    }
    
    # 1. 独立样本t检验（假设方差不等）
    t_stat, t_pvalue = stats.ttest_ind(selected, random, equal_var=False)
    results['t_statistic'] = t_stat
    results['t_pvalue'] = t_pvalue
    
    # 2. Mann-Whitney U检验（非参数检验，不假设正态分布）
    u_stat, u_pvalue = stats.mannwhitneyu(selected, random, alternative='greater')
    results['u_statistic'] = u_stat
    results['u_pvalue'] = u_pvalue
    
    # 3. Cohen's d 效应量
    pooled_std = np.sqrt((results['std_selected']**2 + results['std_random']**2) / 2)
    if pooled_std > 0:
        cohens_d = (results['mean_selected'] - results['mean_random']) / pooled_std
    else:
        cohens_d = 0
    results['cohens_d'] = cohens_d
    
    # 4. 效应量解释
    if abs(cohens_d) < 0.2:
        effect_size = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_size = "small"
    elif abs(cohens_d) < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"
    results['effect_size_interpretation'] = effect_size
    
    # 5. 单样本t检验：LLM选择的百分比是否显著高于随机期望
    # 随机期望 = 100% / 变量数量，但这里我们用实际的随机基线均值
    t_one, p_one = stats.ttest_1samp(selected, results['mean_random'])
    results['one_sample_t'] = t_one
    results['one_sample_p'] = p_one
    
    return results


def analyze_by_rank(results):
    """
    按排名分析：LLM选择的变量在出现次数排名中的位置
    
    返回:
    - ranks_first: 第一次选择的排名列表
    - ranks_second: 第二次选择的排名列表
    """
    ranks_first = []
    ranks_second = []
    
    for result in results:
        all_var_counts = result.get('all_original_var_counts', {})
        if not all_var_counts:
            continue
        
        # 按出现次数降序排序
        sorted_vars = sorted(all_var_counts.items(), key=lambda x: x[1], reverse=True)
        var_to_rank = {var: rank + 1 for rank, (var, _) in enumerate(sorted_vars)}
        
        # 第一次选择的排名
        first_var = result.get('first_selected_original_var')
        if first_var and first_var in var_to_rank:
            ranks_first.append(var_to_rank[first_var])
        
        # 第二次选择的排名
        second_var = result.get('second_selected_original_var')
        if second_var and second_var in var_to_rank:
            ranks_second.append(var_to_rank[second_var])
    
    return ranks_first, ranks_second


def generate_visualizations(selected_first, selected_second, random_baseline, 
                           ranks_first, ranks_second, results):
    """生成可视化图表"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 百分比分布对比 - 第一次选择
    ax1 = axes[0, 0]
    ax1.hist(selected_first, bins=50, alpha=0.7, label='LLM First Selection', 
             color='blue', density=True)
    ax1.hist(random_baseline, bins=50, alpha=0.5, label='Random Baseline', 
             color='gray', density=True)
    ax1.axvline(np.mean(selected_first), color='blue', linestyle='--', 
                label=f'LLM Mean: {np.mean(selected_first):.2f}%')
    ax1.axvline(np.mean(random_baseline), color='gray', linestyle='--', 
                label=f'Random Mean: {np.mean(random_baseline):.2f}%')
    ax1.set_xlabel('Variable Occurrence Percentage (%)')
    ax1.set_ylabel('Density')
    ax1.set_title('First Selection: Percentage Distribution')
    ax1.legend()
    
    # 2. 百分比分布对比 - 第二次选择
    ax2 = axes[0, 1]
    ax2.hist(selected_second, bins=50, alpha=0.7, label='LLM Second Selection', 
             color='green', density=True)
    ax2.hist(random_baseline, bins=50, alpha=0.5, label='Random Baseline', 
             color='gray', density=True)
    ax2.axvline(np.mean(selected_second), color='green', linestyle='--', 
                label=f'LLM Mean: {np.mean(selected_second):.2f}%')
    ax2.axvline(np.mean(random_baseline), color='gray', linestyle='--', 
                label=f'Random Mean: {np.mean(random_baseline):.2f}%')
    ax2.set_xlabel('Variable Occurrence Percentage (%)')
    ax2.set_ylabel('Density')
    ax2.set_title('Second Selection: Percentage Distribution')
    ax2.legend()
    
    # 3. 箱线图对比
    ax3 = axes[0, 2]
    box_data = [selected_first, selected_second, random_baseline]
    bp = ax3.boxplot(box_data, labels=['1st Selection', '2nd Selection', 'Random'])
    ax3.set_ylabel('Variable Occurrence Percentage (%)')
    ax3.set_title('Percentage Comparison (Box Plot)')
    
    # 添加均值点
    means = [np.mean(d) for d in box_data]
    ax3.scatter([1, 2, 3], means, color='red', marker='D', s=50, zorder=3, label='Mean')
    ax3.legend()
    
    # 4. 排名分布 - 第一次选择
    ax4 = axes[1, 0]
    max_rank = max(max(ranks_first) if ranks_first else 1, 
                   max(ranks_second) if ranks_second else 1)
    bins = range(1, min(max_rank + 2, 52))  # 最多显示前50名
    ax4.hist(ranks_first, bins=bins, alpha=0.7, color='blue', edgecolor='black')
    ax4.axvline(np.mean(ranks_first), color='red', linestyle='--', 
                label=f'Mean Rank: {np.mean(ranks_first):.1f}')
    ax4.axvline(np.median(ranks_first), color='orange', linestyle='--', 
                label=f'Median Rank: {np.median(ranks_first):.1f}')
    ax4.set_xlabel('Rank (1 = Most Frequent)')
    ax4.set_ylabel('Count')
    ax4.set_title('First Selection: Rank Distribution')
    ax4.set_xlim(0, 51)
    ax4.legend()
    
    # 5. 排名分布 - 第二次选择
    ax5 = axes[1, 1]
    ax5.hist(ranks_second, bins=bins, alpha=0.7, color='green', edgecolor='black')
    ax5.axvline(np.mean(ranks_second), color='red', linestyle='--', 
                label=f'Mean Rank: {np.mean(ranks_second):.1f}')
    ax5.axvline(np.median(ranks_second), color='orange', linestyle='--', 
                label=f'Median Rank: {np.median(ranks_second):.1f}')
    ax5.set_xlabel('Rank (1 = Most Frequent)')
    ax5.set_ylabel('Count')
    ax5.set_title('Second Selection: Rank Distribution')
    ax5.set_xlim(0, 51)
    ax5.legend()
    
    # 6. 累积分布函数 (CDF)
    ax6 = axes[1, 2]
    
    # 计算CDF
    sorted_first = np.sort(selected_first)
    cdf_first = np.arange(1, len(sorted_first) + 1) / len(sorted_first)
    
    sorted_second = np.sort(selected_second)
    cdf_second = np.arange(1, len(sorted_second) + 1) / len(sorted_second)
    
    sorted_random = np.sort(random_baseline)
    cdf_random = np.arange(1, len(sorted_random) + 1) / len(sorted_random)
    
    ax6.plot(sorted_first, cdf_first, label='1st Selection', color='blue')
    ax6.plot(sorted_second, cdf_second, label='2nd Selection', color='green')
    ax6.plot(sorted_random, cdf_random, label='Random Baseline', color='gray', linestyle='--')
    ax6.set_xlabel('Variable Occurrence Percentage (%)')
    ax6.set_ylabel('Cumulative Probability')
    ax6.set_title('Cumulative Distribution Function (CDF)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'bias_analysis_percentage.png'), dpi=150)
    plt.close()
    
    print(f"可视化图表已保存到: {os.path.join(OUTPUT_DIR, 'bias_analysis_percentage.png')}")


def generate_report(stats_first, stats_second, ranks_first, ranks_second, 
                   selected_first, selected_second, random_baseline):
    """生成详细分析报告"""
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("LLM变量选择偏向性分析报告 - 基于出现次数百分比")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # 方法说明
    report_lines.append("## 分析方法")
    report_lines.append("-" * 40)
    report_lines.append("1. 将变量出现次数转换为百分比：")
    report_lines.append("   百分比 = (该变量出现次数 / 所有变量总出现次数) × 100%")
    report_lines.append("2. 与随机选择基线对比（随机基线 = 所有变量的平均百分比）")
    report_lines.append("3. 使用统计检验证明显著性")
    report_lines.append("")
    
    # 第一次选择统计
    report_lines.append("=" * 80)
    report_lines.append("## 第一次选择分析")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"样本数量: {stats_first['n_selected']}")
    report_lines.append(f"LLM选择的变量平均百分比: {stats_first['mean_selected']:.4f}%")
    report_lines.append(f"随机基线平均百分比: {stats_first['mean_random']:.4f}%")
    report_lines.append(f"差异: +{stats_first['mean_selected'] - stats_first['mean_random']:.4f}%")
    report_lines.append(f"倍数: {stats_first['mean_selected'] / stats_first['mean_random']:.2f}x")
    report_lines.append("")
    report_lines.append("统计检验结果:")
    report_lines.append(f"  - t检验: t={stats_first['t_statistic']:.4f}, p={stats_first['t_pvalue']:.2e}")
    report_lines.append(f"  - Mann-Whitney U检验: U={stats_first['u_statistic']:.0f}, p={stats_first['u_pvalue']:.2e}")
    report_lines.append(f"  - Cohen's d效应量: {stats_first['cohens_d']:.4f} ({stats_first['effect_size_interpretation']})")
    report_lines.append(f"  - 单样本t检验: t={stats_first['one_sample_t']:.4f}, p={stats_first['one_sample_p']:.2e}")
    report_lines.append("")
    
    # 排名分析
    report_lines.append("排名分析 (1=最高频):")
    report_lines.append(f"  - 平均排名: {np.mean(ranks_first):.2f}")
    report_lines.append(f"  - 中位数排名: {np.median(ranks_first):.2f}")
    report_lines.append(f"  - 选择排名第1的比例: {sum(1 for r in ranks_first if r == 1) / len(ranks_first) * 100:.2f}%")
    report_lines.append(f"  - 选择前3名的比例: {sum(1 for r in ranks_first if r <= 3) / len(ranks_first) * 100:.2f}%")
    report_lines.append(f"  - 选择前10名的比例: {sum(1 for r in ranks_first if r <= 10) / len(ranks_first) * 100:.2f}%")
    report_lines.append("")
    
    # 第二次选择统计
    report_lines.append("=" * 80)
    report_lines.append("## 第二次选择分析")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"样本数量: {stats_second['n_selected']}")
    report_lines.append(f"LLM选择的变量平均百分比: {stats_second['mean_selected']:.4f}%")
    report_lines.append(f"随机基线平均百分比: {stats_second['mean_random']:.4f}%")
    report_lines.append(f"差异: +{stats_second['mean_selected'] - stats_second['mean_random']:.4f}%")
    report_lines.append(f"倍数: {stats_second['mean_selected'] / stats_second['mean_random']:.2f}x")
    report_lines.append("")
    report_lines.append("统计检验结果:")
    report_lines.append(f"  - t检验: t={stats_second['t_statistic']:.4f}, p={stats_second['t_pvalue']:.2e}")
    report_lines.append(f"  - Mann-Whitney U检验: U={stats_second['u_statistic']:.0f}, p={stats_second['u_pvalue']:.2e}")
    report_lines.append(f"  - Cohen's d效应量: {stats_second['cohens_d']:.4f} ({stats_second['effect_size_interpretation']})")
    report_lines.append(f"  - 单样本t检验: t={stats_second['one_sample_t']:.4f}, p={stats_second['one_sample_p']:.2e}")
    report_lines.append("")
    
    # 排名分析
    report_lines.append("排名分析 (1=最高频):")
    report_lines.append(f"  - 平均排名: {np.mean(ranks_second):.2f}")
    report_lines.append(f"  - 中位数排名: {np.median(ranks_second):.2f}")
    report_lines.append(f"  - 选择排名第1的比例: {sum(1 for r in ranks_second if r == 1) / len(ranks_second) * 100:.2f}%")
    report_lines.append(f"  - 选择前3名的比例: {sum(1 for r in ranks_second if r <= 3) / len(ranks_second) * 100:.2f}%")
    report_lines.append(f"  - 选择前10名的比例: {sum(1 for r in ranks_second if r <= 10) / len(ranks_second) * 100:.2f}%")
    report_lines.append("")
    
    # 结论
    report_lines.append("=" * 80)
    report_lines.append("## 结论")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # 判断是否显著
    significant_first = stats_first['t_pvalue'] < 0.05 and stats_first['u_pvalue'] < 0.05
    significant_second = stats_second['t_pvalue'] < 0.05 and stats_second['u_pvalue'] < 0.05
    
    if significant_first:
        report_lines.append("### 第一次选择:")
        report_lines.append(f"✓ LLM选择的变量出现百分比显著高于随机基线 (p < 0.05)")
        report_lines.append(f"  - LLM选择的变量平均占比是随机基线的 {stats_first['mean_selected'] / stats_first['mean_random']:.2f} 倍")
        report_lines.append(f"  - 效应量为 {stats_first['effect_size_interpretation']} (Cohen's d = {stats_first['cohens_d']:.4f})")
    else:
        report_lines.append("### 第一次选择:")
        report_lines.append("✗ 未发现显著偏向性 (p >= 0.05)")
    
    report_lines.append("")
    
    if significant_second:
        report_lines.append("### 第二次选择:")
        report_lines.append(f"✓ LLM选择的变量出现百分比显著高于随机基线 (p < 0.05)")
        report_lines.append(f"  - LLM选择的变量平均占比是随机基线的 {stats_second['mean_selected'] / stats_second['mean_random']:.2f} 倍")
        report_lines.append(f"  - 效应量为 {stats_second['effect_size_interpretation']} (Cohen's d = {stats_second['cohens_d']:.4f})")
    else:
        report_lines.append("### 第二次选择:")
        report_lines.append("✗ 未发现显著偏向性 (p >= 0.05)")
    
    report_lines.append("")
    report_lines.append("### 总体结论:")
    if significant_first and significant_second:
        report_lines.append("LLM在选择变量时存在显著的偏向性，倾向于选择出现次数百分比较高的变量。")
        report_lines.append("这一偏向性在第一次和第二次选择中都得到了统计学验证。")
    elif significant_first or significant_second:
        report_lines.append("LLM在部分选择中存在偏向性，但不是完全一致的。")
    else:
        report_lines.append("未发现LLM选择变量时存在显著的出现次数偏向性。")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # 保存报告
    report_text = '\n'.join(report_lines)
    report_file = os.path.join(OUTPUT_DIR, 'bias_analysis_percentage_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n报告已保存到: {report_file}")
    
    return report_text


def main():
    """主函数"""
    print("加载实验结果...")
    results = load_results()
    print(f"共加载 {len(results)} 条记录")
    
    print("\n计算百分比...")
    selected_first, selected_second, random_baseline = analyze_with_percentage(results)
    print(f"第一次选择: {len(selected_first)} 条")
    print(f"第二次选择: {len(selected_second)} 条")
    print(f"随机基线: {len(random_baseline)} 条")
    
    print("\n执行统计检验...")
    stats_first = statistical_tests(selected_first, random_baseline, "First Selection")
    stats_second = statistical_tests(selected_second, random_baseline, "Second Selection")
    
    print("\n分析排名...")
    ranks_first, ranks_second = analyze_by_rank(results)
    
    print("\n生成可视化...")
    generate_visualizations(selected_first, selected_second, random_baseline,
                           ranks_first, ranks_second, results)
    
    print("\n生成报告...")
    generate_report(stats_first, stats_second, ranks_first, ranks_second,
                   selected_first, selected_second, random_baseline)


if __name__ == '__main__':
    main()
