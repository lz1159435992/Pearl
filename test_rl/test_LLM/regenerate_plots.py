"""
重新生成可视化图表和分析报告
修复中文字体显示问题，并进行详细的相关性分析
"""
import json
import os
import sys
import numpy as np
from scipy import stats

try:
    import matplotlib.pyplot as plt
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    HAS_MATPLOTLIB = True
except ImportError:
    print("matplotlib not available")
    HAS_MATPLOTLIB = False


def load_results(results_file):
    """加载实验结果"""
    print(f"正在加载结果文件: {results_file}")
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    print(f"加载了 {len(results)} 个结果")
    return results


def analyze_and_visualize(results, output_dir):
    """分析结果并生成可视化图表"""
    if not HAS_MATPLOTLIB:
        print("matplotlib不可用，无法生成图表")
        return
    
    # 收集数据（使用百分比而不是绝对次数）
    first_selected_percentages = []  # 第一次选择的变量在原文中的出现次数百分比
    second_selected_percentages = []  # 第二次选择的变量在原文中的出现次数百分比
    first_ranks = []  # 第一次选择的变量在所有变量中的排名（1=最高频）
    second_ranks = []  # 第二次选择的变量在所有变量中的排名
    
    for result in results:
        all_counts_dict = result.get('all_original_var_counts', {})
        if not all_counts_dict:
            continue
        
        # 计算所有变量的总出现次数
        total_counts = sum(all_counts_dict.values())
        if total_counts == 0:
            continue
        
        # 获取所有变量的出现次数并排序
        all_counts = sorted(all_counts_dict.values(), reverse=True)
        if len(all_counts) == 0:
            continue
        
        # 第一次选择
        first_count = result.get('first_selected_original_var_count')
        if first_count is not None:
            # 计算百分比
            first_percentage = (first_count / total_counts) * 100
            first_selected_percentages.append(first_percentage)
            rank = sum(1 for c in all_counts if c > first_count) + 1
            first_ranks.append(rank)
        
        # 第二次选择
        second_count = result.get('second_selected_original_var_count')
        if second_count is not None:
            # 计算百分比
            second_percentage = (second_count / total_counts) * 100
            second_selected_percentages.append(second_percentage)
            rank = sum(1 for c in all_counts if c > second_count) + 1
            second_ranks.append(rank)
    
    print(f"第一次选择有效数据: {len(first_selected_percentages)}")
    print(f"第二次选择有效数据: {len(second_selected_percentages)}")
    
    if not first_selected_percentages and not second_selected_percentages:
        print("没有足够的数据生成图表")
        return
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('LLM变量选择与变量出现次数百分比的关系分析', fontsize=16, fontweight='bold')
    
    # 1. 第一次选择：变量出现次数百分比分布直方图
    ax1 = axes[0, 0]
    if first_selected_percentages:
        ax1.hist(first_selected_percentages, bins=30, alpha=0.7, color='#1f77b4', edgecolor='black')
        ax1.set_xlabel('变量出现次数百分比 (%)', fontsize=11)
        ax1.set_ylabel('选择频次', fontsize=11)
        ax1.set_title('第一次选择：变量出现次数百分比分布', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        mean_first = np.mean(first_selected_percentages)
        ax1.axvline(mean_first, color='red', linestyle='--', 
                   label=f'平均值: {mean_first:.2f}%')
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('第一次选择：变量出现次数百分比分布', fontsize=12)
    
    # 2. 第二次选择：变量出现次数百分比分布直方图
    ax2 = axes[0, 1]
    if second_selected_percentages:
        ax2.hist(second_selected_percentages, bins=30, alpha=0.7, color='#ff7f0e', edgecolor='black')
        ax2.set_xlabel('变量出现次数百分比 (%)', fontsize=11)
        ax2.set_ylabel('选择频次', fontsize=11)
        ax2.set_title('第二次选择：变量出现次数百分比分布', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        mean_second = np.mean(second_selected_percentages)
        ax2.axvline(mean_second, color='red', linestyle='--', 
                   label=f'平均值: {mean_second:.2f}%')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('第二次选择：变量出现次数百分比分布', fontsize=12)
    
    # 3. 第一次 vs 第二次选择的箱线图比较
    ax3 = axes[1, 0]
    if first_selected_percentages and second_selected_percentages:
        box_data = [first_selected_percentages, second_selected_percentages]
        bp = ax3.boxplot(box_data, labels=['第一次选择', '第二次选择'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#1f77b4')
        bp['boxes'][1].set_facecolor('#ff7f0e')
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_alpha(0.7)
        ax3.set_ylabel('变量出现次数百分比 (%)', fontsize=11)
        ax3.set_title('第一次 vs 第二次选择：出现次数百分比比较（箱线图）', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
    else:
        ax3.text(0.5, 0.5, '数据不足', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('第一次 vs 第二次选择：出现次数百分比比较', fontsize=12)
    
    # 4. 散点图：第一次选择 vs 第二次选择（如果有配对数据）
    ax4 = axes[1, 1]
    paired_first = []
    paired_second = []
    for result in results:
        all_counts_dict = result.get('all_original_var_counts', {})
        if not all_counts_dict:
            continue
        total_counts = sum(all_counts_dict.values())
        if total_counts == 0:
            continue
        
        first_count = result.get('first_selected_original_var_count')
        second_count = result.get('second_selected_original_var_count')
        if first_count is not None and second_count is not None:
            first_percentage = (first_count / total_counts) * 100
            second_percentage = (second_count / total_counts) * 100
            paired_first.append(first_percentage)
            paired_second.append(second_percentage)
    
    if paired_first and paired_second:
        ax4.scatter(paired_first, paired_second, alpha=0.5, s=50, color='green')
        ax4.set_xlabel('第一次选择变量的出现次数百分比 (%)', fontsize=11)
        ax4.set_ylabel('第二次选择变量的出现次数百分比 (%)', fontsize=11)
        ax4.set_title('第一次 vs 第二次选择：出现次数百分比相关性', fontsize=12, fontweight='bold')
        
        # 添加对角线（y=x）
        max_val = max(max(paired_first), max(paired_second))
        ax4.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='y=x')
        
        # 计算相关系数
        if len(paired_first) > 1:
            corr = np.corrcoef(paired_first, paired_second)[0, 1]
            ax4.text(0.05, 0.95, f'相关系数: {corr:.3f}', transform=ax4.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, '配对数据不足', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('第一次 vs 第二次选择：出现次数百分比相关性', fontsize=12)
    
    plt.tight_layout()
    
    # 保存图表
    plot_file = os.path.join(output_dir, 'variable_selection_analysis.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"可视化图表已保存到: {plot_file}")
    
    plt.close()
    
    # 生成详细分析报告
    generate_detailed_analysis_report(
        first_selected_percentages, second_selected_percentages,
        first_ranks, second_ranks,
        paired_first, paired_second,
        output_dir
    )


def generate_detailed_analysis_report(first_percentages, second_percentages, first_ranks, second_ranks,
                                     paired_first, paired_second, output_dir):
    """生成详细的分析报告（使用百分比）"""
    stats_lines = []
    stats_lines.append("="*80)
    stats_lines.append("变量选择统计分析摘要（基于出现次数百分比）")
    stats_lines.append("="*80)
    stats_lines.append("")
    stats_lines.append("注意：使用百分比可以消除不同大小约束文件的影响，更公平地比较结果")
    stats_lines.append("")
    
    # 第一次选择统计
    if first_percentages:
        stats_lines.append("第一次选择:")
        stats_lines.append(f"  有效选择数: {len(first_percentages)}")
        stats_lines.append(f"  平均出现次数百分比: {np.mean(first_percentages):.4f}%")
        stats_lines.append(f"  中位数出现次数百分比: {np.median(first_percentages):.4f}%")
        stats_lines.append(f"  标准差: {np.std(first_percentages):.4f}%")
        stats_lines.append(f"  最小值: {np.min(first_percentages):.4f}%")
        stats_lines.append(f"  最大值: {np.max(first_percentages):.4f}%")
        stats_lines.append("")
    
    # 第二次选择统计
    if second_percentages:
        stats_lines.append("第二次选择:")
        stats_lines.append(f"  有效选择数: {len(second_percentages)}")
        stats_lines.append(f"  平均出现次数百分比: {np.mean(second_percentages):.4f}%")
        stats_lines.append(f"  中位数出现次数百分比: {np.median(second_percentages):.4f}%")
        stats_lines.append(f"  标准差: {np.std(second_percentages):.4f}%")
        stats_lines.append(f"  最小值: {np.min(second_percentages):.4f}%")
        stats_lines.append(f"  最大值: {np.max(second_percentages):.4f}%")
        stats_lines.append("")
    
    # 排名分析
    if first_ranks:
        stats_lines.append("第一次选择排名分析 (1=最高频):")
        stats_lines.append(f"  平均排名: {np.mean(first_ranks):.2f}")
        stats_lines.append(f"  中位数排名: {np.median(first_ranks):.2f}")
        stats_lines.append(f"  标准差: {np.std(first_ranks):.2f}")
        top1_ratio = sum(1 for r in first_ranks if r == 1) / len(first_ranks) * 100
        top3_ratio = sum(1 for r in first_ranks if r <= 3) / len(first_ranks) * 100
        top10_ratio = sum(1 for r in first_ranks if r <= 10) / len(first_ranks) * 100
        stats_lines.append(f"  选择最高频变量 (排名=1) 的比例: {top1_ratio:.2f}%")
        stats_lines.append(f"  选择前3高频变量 (排名<=3) 的比例: {top3_ratio:.2f}%")
        stats_lines.append(f"  选择前10高频变量 (排名<=10) 的比例: {top10_ratio:.2f}%")
        stats_lines.append("")
    
    if second_ranks:
        stats_lines.append("第二次选择排名分析 (1=最高频):")
        stats_lines.append(f"  平均排名: {np.mean(second_ranks):.2f}")
        stats_lines.append(f"  中位数排名: {np.median(second_ranks):.2f}")
        stats_lines.append(f"  标准差: {np.std(second_ranks):.2f}")
        top1_ratio = sum(1 for r in second_ranks if r == 1) / len(second_ranks) * 100
        top3_ratio = sum(1 for r in second_ranks if r <= 3) / len(second_ranks) * 100
        top10_ratio = sum(1 for r in second_ranks if r <= 10) / len(second_ranks) * 100
        stats_lines.append(f"  选择最高频变量 (排名=1) 的比例: {top1_ratio:.2f}%")
        stats_lines.append(f"  选择前3高频变量 (排名<=3) 的比例: {top3_ratio:.2f}%")
        stats_lines.append(f"  选择前10高频变量 (排名<=10) 的比例: {top10_ratio:.2f}%")
        stats_lines.append("")
    
    # 配对数据相关性
    if paired_first and paired_second and len(paired_first) > 1:
        corr = np.corrcoef(paired_first, paired_second)[0, 1]
        stats_lines.append("第一次 vs 第二次选择:")
        stats_lines.append(f"  配对数据数: {len(paired_first)}")
        stats_lines.append(f"  相关系数: {corr:.4f}")
        if len(paired_first) > 3:
            corr_stat, corr_p = stats.pearsonr(paired_first, paired_second)
            stats_lines.append(f"  统计显著性 (p-value): {corr_p:.4f}")
            stats_lines.append(f"  是否显著相关 (p<0.05): {'是 ✓' if corr_p < 0.05 else '否 ✗'}")
        stats_lines.append("")
    
    # 总体结论
    stats_lines.append("="*80)
    stats_lines.append("总体结论")
    stats_lines.append("="*80)
    
    if first_ranks:
        avg_rank_first = np.mean(first_ranks)
        median_rank_first = np.median(first_ranks)
        top3_ratio_first = sum(1 for r in first_ranks if r <= 3) / len(first_ranks)
        avg_percentage_first = np.mean(first_percentages) if first_percentages else 0
        stats_lines.append(f"\n第一次选择:")
        stats_lines.append(f"  - 平均出现次数百分比: {avg_percentage_first:.4f}%")
        stats_lines.append(f"  - 平均排名: {avg_rank_first:.2f}, 中位数排名: {median_rank_first:.1f} (1=最高频)")
        stats_lines.append(f"  - 前3高频比例: {top3_ratio_first*100:.1f}%")
        if median_rank_first < 50:
            stats_lines.append(f"  - 结论: LLM倾向于选择中等偏高频的变量")
        else:
            stats_lines.append(f"  - 结论: LLM的选择与变量频率相关性较弱，倾向于选择中低频变量")
        stats_lines.append("")
    
    if second_ranks:
        avg_rank_second = np.mean(second_ranks)
        median_rank_second = np.median(second_ranks)
        top3_ratio_second = sum(1 for r in second_ranks if r <= 3) / len(second_ranks)
        avg_percentage_second = np.mean(second_percentages) if second_percentages else 0
        stats_lines.append(f"第二次选择:")
        stats_lines.append(f"  - 平均出现次数百分比: {avg_percentage_second:.4f}%")
        stats_lines.append(f"  - 平均排名: {avg_rank_second:.2f}, 中位数排名: {median_rank_second:.1f} (1=最高频)")
        stats_lines.append(f"  - 前3高频比例: {top3_ratio_second*100:.1f}%")
        if median_rank_second < 50:
            stats_lines.append(f"  - 结论: LLM倾向于选择中等偏高频的变量")
        else:
            stats_lines.append(f"  - 结论: LLM的选择与变量频率相关性较弱，倾向于选择中低频变量")
    
    stats_lines.append("\n" + "="*80)
    stats_lines.append("相关性分析结论")
    stats_lines.append("="*80)
    
    stats_lines.append("\n" + "="*80)
    stats_lines.append("相关性分析结论")
    stats_lines.append("="*80)
    
    if first_percentages and second_percentages:
        stats_lines.append(f"\n基于百分比分析:")
        stats_lines.append(f"  - 第一次选择平均百分比: {np.mean(first_percentages):.4f}%")
        stats_lines.append(f"  - 第二次选择平均百分比: {np.mean(second_percentages):.4f}%")
        if np.mean(first_percentages) > np.mean(second_percentages):
            stats_lines.append(f"  - 第一次选择的变量平均百分比更高")
        elif np.mean(second_percentages) > np.mean(first_percentages):
            stats_lines.append(f"  - 第二次选择的变量平均百分比更高")
        else:
            stats_lines.append(f"  - 两次选择的变量平均百分比相近")
        
        # 计算期望百分比（如果完全随机选择）
        # 对于N个变量，每个变量的期望百分比约为 100/N%
        # 但实际分布不均匀，需要基于实际数据
        stats_lines.append(f"\n  - 使用百分比可以消除文件大小的影响，更公平地比较不同约束文件中的选择行为")
    
    stats_file = os.path.join(output_dir, 'selection_statistics.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(stats_lines))
    print(f"统计摘要已保存到: {stats_file}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_file = os.path.join(script_dir, 'output', 'experiment_results.json')
    output_dir = os.path.join(script_dir, 'output')
    
    if not os.path.exists(results_file):
        print(f"错误: 结果文件不存在: {results_file}")
        sys.exit(1)
    
    results = load_results(results_file)
    analyze_and_visualize(results, output_dir)
    print("\n分析完成！")

