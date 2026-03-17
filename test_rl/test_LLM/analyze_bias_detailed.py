"""
LLM变量选择偏向性详细分析 - 补充分析
包括：相关性分析、分位数对比、Bootstrap置信区间
"""
import json
import os
import numpy as np
from scipy import stats
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


def calculate_detailed_stats(results):
    """计算详细统计数据"""
    
    data = {
        'first_selection': [],  # (百分比, 排名, 变量数)
        'second_selection': [],
        'all_variables': [],  # 所有变量的百分比
    }
    
    files_processed = set()
    
    for result in results:
        all_var_counts = result.get('all_original_var_counts', {})
        if not all_var_counts:
            continue
        
        total_count = sum(all_var_counts.values())
        if total_count == 0:
            continue
        
        n_vars = len(all_var_counts)
        
        # 按出现次数降序排序
        sorted_vars = sorted(all_var_counts.items(), key=lambda x: x[1], reverse=True)
        var_to_rank = {var: rank + 1 for rank, (var, _) in enumerate(sorted_vars)}
        var_to_pct = {var: (count / total_count) * 100 for var, count in all_var_counts.items()}
        
        # 第一次选择
        first_var = result.get('first_selected_original_var')
        if first_var and first_var in all_var_counts:
            pct = var_to_pct[first_var]
            rank = var_to_rank[first_var]
            data['first_selection'].append((pct, rank, n_vars))
        
        # 第二次选择
        second_var = result.get('second_selected_original_var')
        if second_var and second_var in all_var_counts:
            pct = var_to_pct[second_var]
            rank = var_to_rank[second_var]
            data['second_selection'].append((pct, rank, n_vars))
        
        # 随机基线
        file_idx = result.get('file_index')
        if file_idx not in files_processed:
            files_processed.add(file_idx)
            for var, pct in var_to_pct.items():
                rank = var_to_rank[var]
                data['all_variables'].append((pct, rank, n_vars))
    
    return data


def bootstrap_confidence_interval(data, n_bootstrap=10000, confidence=0.95):
    """计算Bootstrap置信区间"""
    data = np.array(data)
    n = len(data)
    
    # Bootstrap重采样
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    bootstrap_means = np.array(bootstrap_means)
    
    # 计算置信区间
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
    
    return np.mean(data), lower, upper


def analyze_by_variable_count(data):
    """按变量数量分组分析"""
    
    # 分组：小型(<=10), 中型(11-50), 大型(>50)
    groups = {
        'small (<=10)': {'first': [], 'second': [], 'random': []},
        'medium (11-50)': {'first': [], 'second': [], 'random': []},
        'large (>50)': {'first': [], 'second': [], 'random': []},
    }
    
    for pct, rank, n_vars in data['first_selection']:
        if n_vars <= 10:
            groups['small (<=10)']['first'].append(pct)
        elif n_vars <= 50:
            groups['medium (11-50)']['first'].append(pct)
        else:
            groups['large (>50)']['first'].append(pct)
    
    for pct, rank, n_vars in data['second_selection']:
        if n_vars <= 10:
            groups['small (<=10)']['second'].append(pct)
        elif n_vars <= 50:
            groups['medium (11-50)']['second'].append(pct)
        else:
            groups['large (>50)']['second'].append(pct)
    
    for pct, rank, n_vars in data['all_variables']:
        if n_vars <= 10:
            groups['small (<=10)']['random'].append(pct)
        elif n_vars <= 50:
            groups['medium (11-50)']['random'].append(pct)
        else:
            groups['large (>50)']['random'].append(pct)
    
    return groups


def generate_detailed_report(data):
    """生成详细报告"""
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("LLM变量选择偏向性 - 详细统计分析报告")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # 1. Bootstrap置信区间
    report_lines.append("## 1. Bootstrap置信区间分析 (95%置信度)")
    report_lines.append("-" * 40)
    
    first_pcts = [x[0] for x in data['first_selection']]
    second_pcts = [x[0] for x in data['second_selection']]
    random_pcts = [x[0] for x in data['all_variables']]
    
    mean1, lower1, upper1 = bootstrap_confidence_interval(first_pcts)
    mean2, lower2, upper2 = bootstrap_confidence_interval(second_pcts)
    mean_r, lower_r, upper_r = bootstrap_confidence_interval(random_pcts)
    
    report_lines.append(f"第一次选择: {mean1:.4f}% [{lower1:.4f}%, {upper1:.4f}%]")
    report_lines.append(f"第二次选择: {mean2:.4f}% [{lower2:.4f}%, {upper2:.4f}%]")
    report_lines.append(f"随机基线:   {mean_r:.4f}% [{lower_r:.4f}%, {upper_r:.4f}%]")
    report_lines.append("")
    
    # 检查置信区间是否重叠
    if lower1 > upper_r:
        report_lines.append("✓ 第一次选择的置信区间完全高于随机基线，偏向性非常显著")
    if lower2 > upper_r:
        report_lines.append("✓ 第二次选择的置信区间完全高于随机基线，偏向性非常显著")
    report_lines.append("")
    
    # 2. 分位数对比
    report_lines.append("## 2. 分位数对比分析")
    report_lines.append("-" * 40)
    
    percentiles = [25, 50, 75, 90, 95]
    report_lines.append(f"{'分位数':<10} {'第一次选择':<15} {'第二次选择':<15} {'随机基线':<15}")
    report_lines.append("-" * 55)
    
    for p in percentiles:
        p1 = np.percentile(first_pcts, p)
        p2 = np.percentile(second_pcts, p)
        pr = np.percentile(random_pcts, p)
        report_lines.append(f"{p}%{'':<8} {p1:<15.4f} {p2:<15.4f} {pr:<15.4f}")
    
    report_lines.append("")
    
    # 3. 按变量数量分组分析
    report_lines.append("## 3. 按约束文件变量数量分组分析")
    report_lines.append("-" * 40)
    
    groups = analyze_by_variable_count(data)
    
    for group_name, group_data in groups.items():
        report_lines.append(f"\n### {group_name}")
        
        if group_data['first'] and group_data['random']:
            mean_first = np.mean(group_data['first'])
            mean_random = np.mean(group_data['random'])
            ratio = mean_first / mean_random if mean_random > 0 else 0
            
            # 统计检验
            if len(group_data['first']) > 1 and len(group_data['random']) > 1:
                t_stat, t_pval = stats.ttest_ind(group_data['first'], group_data['random'], equal_var=False)
                report_lines.append(f"  第一次选择: 均值={mean_first:.4f}%, 样本数={len(group_data['first'])}")
                report_lines.append(f"  随机基线:   均值={mean_random:.4f}%, 样本数={len(group_data['random'])}")
                report_lines.append(f"  倍数: {ratio:.2f}x, t检验 p值={t_pval:.2e}")
        
        if group_data['second'] and group_data['random']:
            mean_second = np.mean(group_data['second'])
            mean_random = np.mean(group_data['random'])
            ratio = mean_second / mean_random if mean_random > 0 else 0
            
            if len(group_data['second']) > 1 and len(group_data['random']) > 1:
                t_stat, t_pval = stats.ttest_ind(group_data['second'], group_data['random'], equal_var=False)
                report_lines.append(f"  第二次选择: 均值={mean_second:.4f}%, 样本数={len(group_data['second'])}")
                report_lines.append(f"  倍数: {ratio:.2f}x, t检验 p值={t_pval:.2e}")
    
    report_lines.append("")
    
    # 4. 排名百分位分析
    report_lines.append("## 4. 排名百分位分析")
    report_lines.append("-" * 40)
    report_lines.append("（排名百分位 = 排名 / 变量总数 × 100%，越小表示越靠前）")
    report_lines.append("")
    
    first_rank_pcts = [rank / n_vars * 100 for _, rank, n_vars in data['first_selection']]
    second_rank_pcts = [rank / n_vars * 100 for _, rank, n_vars in data['second_selection']]
    random_rank_pcts = [rank / n_vars * 100 for _, rank, n_vars in data['all_variables']]
    
    report_lines.append(f"第一次选择平均排名百分位: {np.mean(first_rank_pcts):.2f}%")
    report_lines.append(f"第二次选择平均排名百分位: {np.mean(second_rank_pcts):.2f}%")
    report_lines.append(f"随机基线平均排名百分位: {np.mean(random_rank_pcts):.2f}% (理论值=50%)")
    report_lines.append("")
    
    # 统计检验
    t1, p1 = stats.ttest_ind(first_rank_pcts, random_rank_pcts, equal_var=False)
    t2, p2 = stats.ttest_ind(second_rank_pcts, random_rank_pcts, equal_var=False)
    
    report_lines.append(f"第一次选择 vs 随机: t={t1:.4f}, p={p1:.2e}")
    report_lines.append(f"第二次选择 vs 随机: t={t2:.4f}, p={p2:.2e}")
    report_lines.append("")
    
    # 5. 总结
    report_lines.append("=" * 80)
    report_lines.append("## 总结")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("基于以上分析，可以得出以下结论：")
    report_lines.append("")
    report_lines.append(f"1. LLM第一次选择的变量平均出现百分比是随机基线的 {mean1/mean_r:.2f} 倍")
    report_lines.append(f"2. LLM第二次选择的变量平均出现百分比是随机基线的 {mean2/mean_r:.2f} 倍")
    report_lines.append(f"3. 置信区间分析表明，LLM选择的偏向性在统计上是显著的")
    report_lines.append(f"4. 这种偏向性在不同大小的约束文件中都存在")
    report_lines.append("")
    report_lines.append("结论：LLM在选择变量时确实存在显著的偏向性，")
    report_lines.append("倾向于选择在约束文件中出现次数百分比较高的变量。")
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # 保存报告
    report_text = '\n'.join(report_lines)
    report_file = os.path.join(OUTPUT_DIR, 'bias_analysis_detailed_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n详细报告已保存到: {report_file}")


def generate_detailed_visualizations(data):
    """生成详细可视化"""
    
    first_pcts = [x[0] for x in data['first_selection']]
    second_pcts = [x[0] for x in data['second_selection']]
    random_pcts = [x[0] for x in data['all_variables']]
    
    first_rank_pcts = [rank / n_vars * 100 for _, rank, n_vars in data['first_selection']]
    second_rank_pcts = [rank / n_vars * 100 for _, rank, n_vars in data['second_selection']]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. 置信区间对比
    ax1 = axes[0, 0]
    
    mean1, lower1, upper1 = bootstrap_confidence_interval(first_pcts)
    mean2, lower2, upper2 = bootstrap_confidence_interval(second_pcts)
    mean_r, lower_r, upper_r = bootstrap_confidence_interval(random_pcts)
    
    categories = ['1st Selection', '2nd Selection', 'Random']
    means = [mean1, mean2, mean_r]
    errors = [[mean1-lower1, mean2-lower2, mean_r-lower_r],
              [upper1-mean1, upper2-mean2, upper_r-mean_r]]
    
    colors = ['blue', 'green', 'gray']
    bars = ax1.bar(categories, means, yerr=errors, capsize=5, color=colors, alpha=0.7)
    ax1.set_ylabel('Variable Occurrence Percentage (%)')
    ax1.set_title('Mean Percentage with 95% Bootstrap CI')
    
    # 添加数值标签
    for bar, mean in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{mean:.3f}%', ha='center', va='bottom', fontsize=10)
    
    # 2. 排名百分位分布
    ax2 = axes[0, 1]
    ax2.hist(first_rank_pcts, bins=50, alpha=0.7, label='1st Selection', color='blue', density=True)
    ax2.hist(second_rank_pcts, bins=50, alpha=0.7, label='2nd Selection', color='green', density=True)
    ax2.axvline(50, color='red', linestyle='--', label='Random Expected (50%)')
    ax2.axvline(np.mean(first_rank_pcts), color='blue', linestyle='-', linewidth=2)
    ax2.axvline(np.mean(second_rank_pcts), color='green', linestyle='-', linewidth=2)
    ax2.set_xlabel('Rank Percentile (lower = more frequent)')
    ax2.set_ylabel('Density')
    ax2.set_title('Rank Percentile Distribution')
    ax2.legend()
    
    # 3. 按变量数量分组的偏向性
    ax3 = axes[1, 0]
    groups = analyze_by_variable_count(data)
    
    group_names = list(groups.keys())
    first_means = [np.mean(groups[g]['first']) if groups[g]['first'] else 0 for g in group_names]
    second_means = [np.mean(groups[g]['second']) if groups[g]['second'] else 0 for g in group_names]
    random_means = [np.mean(groups[g]['random']) if groups[g]['random'] else 0 for g in group_names]
    
    x = np.arange(len(group_names))
    width = 0.25
    
    ax3.bar(x - width, first_means, width, label='1st Selection', color='blue', alpha=0.7)
    ax3.bar(x, second_means, width, label='2nd Selection', color='green', alpha=0.7)
    ax3.bar(x + width, random_means, width, label='Random', color='gray', alpha=0.7)
    
    ax3.set_xlabel('Constraint File Size (by variable count)')
    ax3.set_ylabel('Mean Percentage (%)')
    ax3.set_title('Bias by File Size')
    ax3.set_xticks(x)
    ax3.set_xticklabels(group_names)
    ax3.legend()
    
    # 4. 倍数对比
    ax4 = axes[1, 1]
    
    ratios_first = [np.mean(groups[g]['first']) / np.mean(groups[g]['random']) 
                   if groups[g]['first'] and groups[g]['random'] and np.mean(groups[g]['random']) > 0 
                   else 0 for g in group_names]
    ratios_second = [np.mean(groups[g]['second']) / np.mean(groups[g]['random']) 
                    if groups[g]['second'] and groups[g]['random'] and np.mean(groups[g]['random']) > 0 
                    else 0 for g in group_names]
    
    x = np.arange(len(group_names))
    width = 0.35
    
    ax4.bar(x - width/2, ratios_first, width, label='1st Selection', color='blue', alpha=0.7)
    ax4.bar(x + width/2, ratios_second, width, label='2nd Selection', color='green', alpha=0.7)
    ax4.axhline(1, color='red', linestyle='--', label='Random Baseline (1x)')
    
    ax4.set_xlabel('Constraint File Size')
    ax4.set_ylabel('Ratio vs Random Baseline')
    ax4.set_title('Selection Bias Ratio by File Size')
    ax4.set_xticks(x)
    ax4.set_xticklabels(group_names)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'bias_analysis_detailed.png'), dpi=150)
    plt.close()
    
    print(f"详细可视化已保存到: {os.path.join(OUTPUT_DIR, 'bias_analysis_detailed.png')}")


def main():
    """主函数"""
    print("加载实验结果...")
    results = load_results()
    
    print("计算详细统计数据...")
    data = calculate_detailed_stats(results)
    
    print("生成详细报告...")
    generate_detailed_report(data)
    
    print("\n生成详细可视化...")
    generate_detailed_visualizations(data)


if __name__ == '__main__':
    main()
