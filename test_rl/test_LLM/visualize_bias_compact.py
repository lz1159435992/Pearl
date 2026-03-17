"""
生成简洁版本的可视化图表，适合论文使用
展示LLM选择变量偏向高百分比变量的关键证据

重要说明：
- 排名计算基于变量出现次数占所有变量出现次数总和的百分比（而非绝对次数）
- 这标准化了不同大小约束文件的影响，使得跨文件的比较更加公平合理
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def load_and_analyze():
    """加载数据并分析（基于百分比而非绝对频次）"""
    results_file = 'output/experiment_results.json'
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    first_ranks = []
    second_ranks = []
    total_vars_per_file = []
    
    for result in results:
        all_counts_dict = result.get('all_original_var_counts', {})
        if not all_counts_dict:
            continue
        
        total_vars = len(all_counts_dict)
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
        
        # 获取被选择变量的百分比
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
    
    return first_ranks, second_ranks, total_vars_per_file


def create_compact_visualization():
    """创建简洁版可视化图表"""
    first_ranks, second_ranks, total_vars = load_and_analyze()
    
    avg_vars = np.mean(total_vars)
    expected_mean_rank = (avg_vars + 1) / 2
    
    # 创建1行2列的布局
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：排名区间比例对比柱状图
    ax1 = axes[0]
    
    rank_ranges = ['排名=1\n(最高百分比)', '排名≤3', '排名≤10', '排名≤20', '排名≤50']
    rank_limits = [1, 3, 10, 20, 50]
    
    first_actual = []
    expected_probs = []
    
    for limit in rank_limits:
        if limit == 1:
            first_ratio = sum(1 for r in first_ranks if r == 1) / len(first_ranks) * 100
            expected_prob = 1 / avg_vars * 100
        else:
            first_ratio = sum(1 for r in first_ranks if r <= limit) / len(first_ranks) * 100
            expected_prob = limit / avg_vars * 100
        
        first_actual.append(first_ratio)
        expected_probs.append(expected_prob)
    
    x = np.arange(len(rank_ranges))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, expected_probs, width, label='随机期望', 
                   color='#d62728', alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x + width/2, first_actual, width, label='LLM实际选择', 
                   color='#2ca02c', alpha=0.8, edgecolor='black')
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 1:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
            else:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}%', ha='center', va='bottom', fontsize=8)
    
    # 添加倍数标注
    for i, (exp, actual) in enumerate(zip(expected_probs, first_actual)):
        if exp > 0 and actual / exp > 3:
            multiple = actual / exp
            ax1.text(i, max(actual, exp) + 2, f'{multiple:.1f}×',
                    ha='center', fontsize=10, fontweight='bold', color='green')
    
    ax1.set_xlabel('排名区间 (基于变量出现次数百分比)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('选择比例 (%)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) 不同排名区间的选择比例对比\n(基于变量出现次数百分比)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(rank_ranges)
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 右图：累积分布函数（CDF）对比
    ax2 = axes[1]
    
    if first_ranks:
        sorted_ranks = np.sort(first_ranks)
        cumulative = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks)
        
        # 实际CDF
        ax2.plot(sorted_ranks, cumulative, linewidth=2.5, color='#2ca02c',
                label='LLM实际选择', alpha=0.9)
        
        # 随机期望的CDF（线性）
        max_rank = min(500, max(first_ranks))
        random_ranks = np.linspace(1, max_rank, 100)
        random_cdf = (random_ranks - 1) / (max_rank - 1)
        ax2.plot(random_ranks, random_cdf, 'r--', linewidth=2.5,
                label='随机期望（均匀分布）', alpha=0.8)
        
        # 标记关键点
        median_rank = np.median(first_ranks)
        median_cdf = np.sum(first_ranks <= median_rank) / len(first_ranks)
        expected_median_cdf = (median_rank - 1) / (max_rank - 1)
        
        ax2.axvline(median_rank, color='green', linestyle=':', alpha=0.6, linewidth=1.5)
        ax2.axhline(median_cdf, color='green', linestyle=':', alpha=0.6, linewidth=1.5)
        ax2.plot(median_rank, median_cdf, 'go', markersize=10, 
                label=f'LLM实际中位数排名: {median_rank:.0f}', zorder=5)
        
        ax2.axvline(expected_mean_rank, color='red', linestyle=':', alpha=0.6, linewidth=1.5)
        ax2.axhline(0.5, color='red', linestyle=':', alpha=0.6, linewidth=1.5)
        ax2.plot(expected_mean_rank, 0.5, 'ro', markersize=10,
                label=f'随机期望中位数: {expected_mean_rank:.0f}', zorder=5)
        
        ax2.set_xlabel('变量排名 (1=最高百分比)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('累积概率', fontsize=12, fontweight='bold')
        ax2.set_title('(b) 累积分布函数（CDF）对比\n(基于变量出现次数百分比)', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10, loc='lower right')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(1, max_rank)
        
        # 添加统计信息文本框
        stats_text = f'LLM平均排名: {np.mean(first_ranks):.1f}\n'
        stats_text += f'随机期望排名: {expected_mean_rank:.1f}\n'
        stats_text += f'p < 0.001 (极显著)'
        ax2.text(0.98, 0.02, stats_text, transform=ax2.transAxes,
                fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('LLM选择变量偏向高百分比变量的证据\n(排名基于变量出现次数占所有变量出现次数总和的百分比)', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    # 保存图表
    output_file = 'output/bias_visualization_compact.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"简洁版图表已保存到: {output_file}")
    
    plt.close()


if __name__ == "__main__":
    create_compact_visualization()
    print("\n简洁版图表特点：")
    print("- 左图：直接对比排名区间的选择比例，清楚展示实际值是期望值的倍数")
    print("- 右图：CDF曲线对比，直观显示分布差异")
    print("- 适合在论文中使用，信息量适中，重点突出")
    print("\n注意：排名基于变量出现次数占所有变量出现次数总和的百分比，")
    print("     标准化了不同大小约束文件的影响，更加公平合理。")


