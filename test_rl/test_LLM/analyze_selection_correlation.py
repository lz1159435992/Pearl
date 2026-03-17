"""
分析第一次选择和第二次选择之间的相关性
"""
import json
import os
import numpy as np
from scipy import stats
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')


def load_results():
    """加载实验结果"""
    results_file = os.path.join(OUTPUT_DIR, 'experiment_results.json')
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_correlation(results):
    """分析第一次和第二次选择的相关性"""
    
    # 收集配对数据
    first_pcts = []  # 第一次选择的百分比
    second_pcts = []  # 第二次选择的百分比
    first_ranks = []  # 第一次选择的排名
    second_ranks = []  # 第二次选择的排名
    
    # 分析选择模式
    same_original_var = 0  # 选择了相同的原始变量（不应该发生）
    first_higher_freq = 0  # 第一次选择的频次更高
    second_higher_freq = 0  # 第二次选择的频次更高
    equal_freq = 0  # 频次相等
    
    # 排名差异
    rank_diffs = []  # 第二次排名 - 第一次排名
    
    # 按文件分组分析一致性
    file_selections = defaultdict(list)  # {file_idx: [(first_orig, second_orig), ...]}
    
    for result in results:
        all_var_counts = result.get('all_original_var_counts', {})
        if not all_var_counts:
            continue
        
        total_count = sum(all_var_counts.values())
        if total_count == 0:
            continue
        
        # 获取第一次和第二次选择
        first_var = result.get('first_selected_original_var')
        second_var = result.get('second_selected_original_var')
        
        if not first_var or not second_var:
            continue
        
        if first_var not in all_var_counts or second_var not in all_var_counts:
            continue
        
        # 计算百分比
        first_pct = (all_var_counts[first_var] / total_count) * 100
        second_pct = (all_var_counts[second_var] / total_count) * 100
        
        first_pcts.append(first_pct)
        second_pcts.append(second_pct)
        
        # 计算排名
        sorted_vars = sorted(all_var_counts.items(), key=lambda x: x[1], reverse=True)
        var_to_rank = {var: rank + 1 for rank, (var, _) in enumerate(sorted_vars)}
        
        first_rank = var_to_rank[first_var]
        second_rank = var_to_rank[second_var]
        
        first_ranks.append(first_rank)
        second_ranks.append(second_rank)
        
        # 分析选择模式
        if first_var == second_var:
            same_original_var += 1
        elif first_pct > second_pct:
            first_higher_freq += 1
        elif second_pct > first_pct:
            second_higher_freq += 1
        else:
            equal_freq += 1
        
        # 排名差异
        rank_diffs.append(second_rank - first_rank)
        
        # 记录文件选择
        file_idx = result.get('file_index')
        file_selections[file_idx].append((first_var, second_var))
    
    return {
        'first_pcts': first_pcts,
        'second_pcts': second_pcts,
        'first_ranks': first_ranks,
        'second_ranks': second_ranks,
        'same_original_var': same_original_var,
        'first_higher_freq': first_higher_freq,
        'second_higher_freq': second_higher_freq,
        'equal_freq': equal_freq,
        'rank_diffs': rank_diffs,
        'file_selections': file_selections,
        'total_pairs': len(first_pcts)
    }


def statistical_analysis(data):
    """统计分析"""
    
    results = {}
    
    # 1. Pearson相关系数（百分比）
    r_pct, p_pct = stats.pearsonr(data['first_pcts'], data['second_pcts'])
    results['pearson_pct'] = {'r': r_pct, 'p': p_pct}
    
    # 2. Spearman秩相关系数（排名）
    r_rank, p_rank = stats.spearmanr(data['first_ranks'], data['second_ranks'])
    results['spearman_rank'] = {'r': r_rank, 'p': p_rank}
    
    # 3. 配对t检验：第一次选择的百分比是否显著高于第二次
    t_stat, t_pval = stats.ttest_rel(data['first_pcts'], data['second_pcts'])
    results['paired_ttest'] = {'t': t_stat, 'p': t_pval}
    
    # 4. Wilcoxon符号秩检验（非参数配对检验）
    w_stat, w_pval = stats.wilcoxon(data['first_pcts'], data['second_pcts'])
    results['wilcoxon'] = {'w': w_stat, 'p': w_pval}
    
    # 5. 排名差异分析
    rank_diffs = np.array(data['rank_diffs'])
    results['rank_diff'] = {
        'mean': np.mean(rank_diffs),
        'median': np.median(rank_diffs),
        'std': np.std(rank_diffs),
        'positive_ratio': np.sum(rank_diffs > 0) / len(rank_diffs),  # 第二次排名更低（频次更低）
        'negative_ratio': np.sum(rank_diffs < 0) / len(rank_diffs),  # 第二次排名更高（频次更高）
        'zero_ratio': np.sum(rank_diffs == 0) / len(rank_diffs)  # 排名相同
    }
    
    return results


def analyze_selection_patterns(data):
    """分析选择模式"""
    
    patterns = {}
    
    # 1. 第一次选择频次更高的比例
    total = data['total_pairs']
    patterns['first_higher'] = data['first_higher_freq'] / total * 100
    patterns['second_higher'] = data['second_higher_freq'] / total * 100
    patterns['equal'] = data['equal_freq'] / total * 100
    patterns['same_var'] = data['same_original_var'] / total * 100
    
    # 2. 分析文件内的选择一致性
    file_consistency = []
    for file_idx, selections in data['file_selections'].items():
        first_vars = [s[0] for s in selections]
        second_vars = [s[1] for s in selections]
        
        # 第一次选择的一致性
        first_unique = len(set(first_vars))
        first_consistency = 1 / first_unique if first_unique > 0 else 0
        
        # 第二次选择的一致性
        second_unique = len(set(second_vars))
        second_consistency = 1 / second_unique if second_unique > 0 else 0
        
        # 第一次和第二次选择的组合一致性
        pairs = [(f, s) for f, s in selections]
        pair_unique = len(set(pairs))
        pair_consistency = 1 / pair_unique if pair_unique > 0 else 0
        
        file_consistency.append({
            'file_idx': file_idx,
            'n_mappings': len(selections),
            'first_unique': first_unique,
            'second_unique': second_unique,
            'pair_unique': pair_unique,
            'first_consistency': first_consistency,
            'second_consistency': second_consistency,
            'pair_consistency': pair_consistency
        })
    
    patterns['file_consistency'] = file_consistency
    
    return patterns


def generate_visualizations(data, stats_results):
    """生成可视化"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 散点图：第一次 vs 第二次选择的百分比
    ax1 = axes[0, 0]
    ax1.scatter(data['first_pcts'], data['second_pcts'], alpha=0.3, s=10)
    ax1.plot([0, max(data['first_pcts'])], [0, max(data['first_pcts'])], 
             'r--', label='y=x (equal)')
    ax1.set_xlabel('First Selection Percentage (%)')
    ax1.set_ylabel('Second Selection Percentage (%)')
    ax1.set_title(f'First vs Second Selection\n(Pearson r={stats_results["pearson_pct"]["r"]:.3f}, p={stats_results["pearson_pct"]["p"]:.2e})')
    ax1.legend()
    
    # 2. 散点图：第一次 vs 第二次选择的排名
    ax2 = axes[0, 1]
    ax2.scatter(data['first_ranks'], data['second_ranks'], alpha=0.3, s=10)
    ax2.plot([0, max(data['first_ranks'])], [0, max(data['first_ranks'])], 
             'r--', label='y=x (equal)')
    ax2.set_xlabel('First Selection Rank (1=highest freq)')
    ax2.set_ylabel('Second Selection Rank')
    ax2.set_title(f'First vs Second Selection Rank\n(Spearman r={stats_results["spearman_rank"]["r"]:.3f}, p={stats_results["spearman_rank"]["p"]:.2e})')
    ax2.legend()
    
    # 3. 排名差异分布
    ax3 = axes[0, 2]
    ax3.hist(data['rank_diffs'], bins=50, alpha=0.7, edgecolor='black')
    ax3.axvline(0, color='red', linestyle='--', label='No difference')
    ax3.axvline(np.mean(data['rank_diffs']), color='blue', linestyle='-', 
                label=f'Mean: {np.mean(data["rank_diffs"]):.1f}')
    ax3.set_xlabel('Rank Difference (2nd - 1st)')
    ax3.set_ylabel('Count')
    ax3.set_title('Rank Difference Distribution\n(Positive = 2nd has lower freq)')
    ax3.legend()
    
    # 4. 选择模式饼图
    ax4 = axes[1, 0]
    labels = ['1st Higher Freq', '2nd Higher Freq', 'Equal Freq', 'Same Var']
    sizes = [data['first_higher_freq'], data['second_higher_freq'], 
             data['equal_freq'], data['same_original_var']]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax4.set_title('Selection Pattern Distribution')
    
    # 5. 百分比差异分布
    ax5 = axes[1, 1]
    pct_diffs = np.array(data['first_pcts']) - np.array(data['second_pcts'])
    ax5.hist(pct_diffs, bins=50, alpha=0.7, edgecolor='black')
    ax5.axvline(0, color='red', linestyle='--', label='No difference')
    ax5.axvline(np.mean(pct_diffs), color='blue', linestyle='-', 
                label=f'Mean: {np.mean(pct_diffs):.3f}%')
    ax5.set_xlabel('Percentage Difference (1st - 2nd)')
    ax5.set_ylabel('Count')
    ax5.set_title(f'Percentage Difference Distribution\n(Paired t-test p={stats_results["paired_ttest"]["p"]:.2e})')
    ax5.legend()
    
    # 6. 累积分布对比
    ax6 = axes[1, 2]
    sorted_first = np.sort(data['first_pcts'])
    sorted_second = np.sort(data['second_pcts'])
    cdf_first = np.arange(1, len(sorted_first) + 1) / len(sorted_first)
    cdf_second = np.arange(1, len(sorted_second) + 1) / len(sorted_second)
    
    ax6.plot(sorted_first, cdf_first, label='First Selection', color='blue')
    ax6.plot(sorted_second, cdf_second, label='Second Selection', color='green')
    ax6.set_xlabel('Variable Occurrence Percentage (%)')
    ax6.set_ylabel('Cumulative Probability')
    ax6.set_title('CDF Comparison: First vs Second Selection')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'selection_correlation_analysis.png'), dpi=150)
    plt.close()
    
    print(f"可视化已保存到: {os.path.join(OUTPUT_DIR, 'selection_correlation_analysis.png')}")


def generate_report(data, stats_results, patterns):
    """生成报告"""
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("第一次选择与第二次选择的相关性分析报告")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # 基本统计
    report_lines.append("## 1. 基本统计")
    report_lines.append("-" * 40)
    report_lines.append(f"配对样本数: {data['total_pairs']}")
    report_lines.append(f"第一次选择平均百分比: {np.mean(data['first_pcts']):.4f}%")
    report_lines.append(f"第二次选择平均百分比: {np.mean(data['second_pcts']):.4f}%")
    report_lines.append(f"差异: {np.mean(data['first_pcts']) - np.mean(data['second_pcts']):.4f}%")
    report_lines.append("")
    
    # 相关性分析
    report_lines.append("## 2. 相关性分析")
    report_lines.append("-" * 40)
    report_lines.append("")
    report_lines.append("### 2.1 Pearson相关系数（百分比）")
    r_pct = stats_results['pearson_pct']
    report_lines.append(f"  r = {r_pct['r']:.4f}, p = {r_pct['p']:.2e}")
    if r_pct['p'] < 0.05:
        if r_pct['r'] > 0.7:
            report_lines.append("  解读: 强正相关，第一次选择高频变量时，第二次也倾向于选择高频变量")
        elif r_pct['r'] > 0.4:
            report_lines.append("  解读: 中等正相关")
        elif r_pct['r'] > 0:
            report_lines.append("  解读: 弱正相关")
        else:
            report_lines.append("  解读: 负相关或无相关")
    else:
        report_lines.append("  解读: 相关性不显著")
    report_lines.append("")
    
    report_lines.append("### 2.2 Spearman秩相关系数（排名）")
    r_rank = stats_results['spearman_rank']
    report_lines.append(f"  r = {r_rank['r']:.4f}, p = {r_rank['p']:.2e}")
    if r_rank['p'] < 0.05:
        if r_rank['r'] > 0.7:
            report_lines.append("  解读: 强正相关，排名具有一致性")
        elif r_rank['r'] > 0.4:
            report_lines.append("  解读: 中等正相关")
        elif r_rank['r'] > 0:
            report_lines.append("  解读: 弱正相关")
    report_lines.append("")
    
    # 配对检验
    report_lines.append("## 3. 配对检验：第一次选择是否显著高于第二次")
    report_lines.append("-" * 40)
    report_lines.append("")
    
    t_test = stats_results['paired_ttest']
    report_lines.append(f"### 3.1 配对t检验")
    report_lines.append(f"  t = {t_test['t']:.4f}, p = {t_test['p']:.2e}")
    if t_test['p'] < 0.05:
        if t_test['t'] > 0:
            report_lines.append("  结论: 第一次选择的变量出现百分比显著高于第二次选择")
        else:
            report_lines.append("  结论: 第二次选择的变量出现百分比显著高于第一次选择")
    else:
        report_lines.append("  结论: 两次选择的变量出现百分比无显著差异")
    report_lines.append("")
    
    wilcoxon = stats_results['wilcoxon']
    report_lines.append(f"### 3.2 Wilcoxon符号秩检验（非参数）")
    report_lines.append(f"  W = {wilcoxon['w']:.0f}, p = {wilcoxon['p']:.2e}")
    report_lines.append("")
    
    # 排名差异分析
    report_lines.append("## 4. 排名差异分析")
    report_lines.append("-" * 40)
    rank_diff = stats_results['rank_diff']
    report_lines.append(f"排名差异 = 第二次排名 - 第一次排名")
    report_lines.append(f"  平均差异: {rank_diff['mean']:.2f}")
    report_lines.append(f"  中位数差异: {rank_diff['median']:.2f}")
    report_lines.append(f"  标准差: {rank_diff['std']:.2f}")
    report_lines.append("")
    report_lines.append(f"  第二次排名更低（频次更低）的比例: {rank_diff['positive_ratio']*100:.1f}%")
    report_lines.append(f"  第二次排名更高（频次更高）的比例: {rank_diff['negative_ratio']*100:.1f}%")
    report_lines.append(f"  排名相同的比例: {rank_diff['zero_ratio']*100:.1f}%")
    report_lines.append("")
    
    # 选择模式分析
    report_lines.append("## 5. 选择模式分析")
    report_lines.append("-" * 40)
    report_lines.append(f"第一次选择频次更高: {patterns['first_higher']:.1f}%")
    report_lines.append(f"第二次选择频次更高: {patterns['second_higher']:.1f}%")
    report_lines.append(f"频次相等: {patterns['equal']:.1f}%")
    report_lines.append(f"选择相同变量: {patterns['same_var']:.1f}%")
    report_lines.append("")
    
    # 文件内一致性分析
    report_lines.append("## 6. 文件内选择一致性分析")
    report_lines.append("-" * 40)
    
    avg_first_unique = np.mean([fc['first_unique'] for fc in patterns['file_consistency']])
    avg_second_unique = np.mean([fc['second_unique'] for fc in patterns['file_consistency']])
    avg_pair_unique = np.mean([fc['pair_unique'] for fc in patterns['file_consistency']])
    
    report_lines.append(f"平均每个文件（50种映射）:")
    report_lines.append(f"  第一次选择的不同原始变量数: {avg_first_unique:.1f}")
    report_lines.append(f"  第二次选择的不同原始变量数: {avg_second_unique:.1f}")
    report_lines.append(f"  (第一次, 第二次)组合的不同数: {avg_pair_unique:.1f}")
    report_lines.append("")
    
    # 结论
    report_lines.append("=" * 80)
    report_lines.append("## 结论")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    r_pct_val = stats_results['pearson_pct']['r']
    if r_pct_val > 0.5:
        report_lines.append(f"1. 第一次和第二次选择存在**中等到强的正相关** (r={r_pct_val:.3f})")
        report_lines.append("   这意味着：当LLM第一次选择高频变量时，第二次也倾向于选择高频变量")
    elif r_pct_val > 0.3:
        report_lines.append(f"1. 第一次和第二次选择存在**弱到中等的正相关** (r={r_pct_val:.3f})")
    else:
        report_lines.append(f"1. 第一次和第二次选择的相关性较弱 (r={r_pct_val:.3f})")
    
    report_lines.append("")
    
    if stats_results['paired_ttest']['p'] < 0.05:
        if stats_results['paired_ttest']['t'] > 0:
            report_lines.append("2. 第一次选择的变量出现百分比**显著高于**第二次选择")
            report_lines.append("   这符合预期：LLM首先选择最重要（高频）的变量，然后选择次重要的")
        else:
            report_lines.append("2. 第二次选择的变量出现百分比显著高于第一次选择")
    else:
        report_lines.append("2. 两次选择的变量出现百分比无显著差异")
    
    report_lines.append("")
    report_lines.append(f"3. 在{patterns['first_higher']:.1f}%的情况下，第一次选择的变量频次更高")
    report_lines.append(f"   在{patterns['second_higher']:.1f}%的情况下，第二次选择的变量频次更高")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # 保存报告
    report_text = '\n'.join(report_lines)
    report_file = os.path.join(OUTPUT_DIR, 'selection_correlation_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n报告已保存到: {report_file}")


def main():
    """主函数"""
    print("加载实验结果...")
    results = load_results()
    
    print("分析相关性...")
    data = analyze_correlation(results)
    
    print("执行统计分析...")
    stats_results = statistical_analysis(data)
    
    print("分析选择模式...")
    patterns = analyze_selection_patterns(data)
    
    print("生成可视化...")
    generate_visualizations(data, stats_results)
    
    print("\n生成报告...")
    generate_report(data, stats_results, patterns)


if __name__ == '__main__':
    main()
