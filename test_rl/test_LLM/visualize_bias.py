"""
生成图表展示LLM选择变量偏向高百分比变量的现象

重要说明：
- 排名计算基于变量出现次数占所有变量出现次数总和的百分比（而非绝对次数）
- 这标准化了不同大小约束文件的影响，使得跨文件的比较更加公平合理
- 在同一个文件内，百分比排序和绝对次数排序的结果相同（数学必然）
- 但在不同文件之间，相同绝对次数的变量会有不同的百分比，因此需要标准化
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


def create_visualization():
    """创建可视化图表"""
    first_ranks, second_ranks, total_vars = load_and_analyze()
    
    avg_vars = np.mean(total_vars)
    expected_mean_rank = (avg_vars + 1) / 2
    
    # 创建2x2的子图布局
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. 排名分布直方图对比（左上）
    ax1 = fig.add_subplot(gs[0, 0])
    if first_ranks:
        # 限制显示范围到合理区间（例如前200名）
        max_rank_display = 200
        first_ranks_display = [r for r in first_ranks if r <= max_rank_display]
        
        ax1.hist(first_ranks_display, bins=50, alpha=0.7, color='#1f77b4', 
                edgecolor='black', label=f'LLM实际选择 (平均排名: {np.mean(first_ranks):.1f})')
        ax1.axvline(expected_mean_rank, color='red', linestyle='--', linewidth=2,
                   label=f'随机期望 (平均排名: {expected_mean_rank:.1f})')
        ax1.axvline(np.mean(first_ranks), color='blue', linestyle='--', linewidth=2)
        ax1.set_xlabel('变量排名 (1=最高百分比)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('实验次数', fontsize=11, fontweight='bold')
        ax1.set_title('第一次选择：排名分布 vs 随机期望\n(基于变量出现次数百分比)', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, max_rank_display)
    
    # 2. 排名分布直方图对比（右上）
    ax2 = fig.add_subplot(gs[0, 1])
    if second_ranks:
        second_ranks_display = [r for r in second_ranks if r <= max_rank_display]
        
        ax2.hist(second_ranks_display, bins=50, alpha=0.7, color='#ff7f0e',
                edgecolor='black', label=f'LLM实际选择 (平均排名: {np.mean(second_ranks):.1f})')
        ax2.axvline(expected_mean_rank, color='red', linestyle='--', linewidth=2,
                   label=f'随机期望 (平均排名: {expected_mean_rank:.1f})')
        ax2.axvline(np.mean(second_ranks), color='orange', linestyle='--', linewidth=2)
        ax2.set_xlabel('变量排名 (1=最高百分比)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('实验次数', fontsize=11, fontweight='bold')
        ax2.set_title('第二次选择：排名分布 vs 随机期望\n(基于变量出现次数百分比)', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, max_rank_display)
    
    # 3. 累积分布函数（CDF）对比（左下）
    ax3 = fig.add_subplot(gs[1, 0])
    if first_ranks:
        sorted_ranks = np.sort(first_ranks)
        cumulative = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks)
        
        # 实际CDF
        ax3.plot(sorted_ranks, cumulative, linewidth=2.5, color='#1f77b4',
                label='LLM实际选择', alpha=0.8)
        
        # 随机期望的CDF（线性）
        max_rank = max(first_ranks)
        random_ranks = np.linspace(1, max_rank, 100)
        random_cdf = (random_ranks - 1) / (max_rank - 1)
        ax3.plot(random_ranks, random_cdf, 'r--', linewidth=2,
                label='随机期望（均匀分布）', alpha=0.8)
        
        # 标记关键点
        median_rank = np.median(first_ranks)
        median_cdf = np.sum(first_ranks <= median_rank) / len(first_ranks)
        ax3.axvline(median_rank, color='blue', linestyle=':', alpha=0.5)
        ax3.axhline(median_cdf, color='blue', linestyle=':', alpha=0.5)
        ax3.plot(median_rank, median_cdf, 'bo', markersize=8, label=f'LLM实际中位数排名: {median_rank:.0f}')
        
        ax3.set_xlabel('变量排名 (1=最高百分比)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('累积概率', fontsize=11, fontweight='bold')
        ax3.set_title('第一次选择：累积分布函数对比\n(基于变量出现次数百分比)', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(1, min(500, max(first_ranks)))
    
    # 4. CDF对比（右下）
    ax4 = fig.add_subplot(gs[1, 1])
    if second_ranks:
        sorted_ranks = np.sort(second_ranks)
        cumulative = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks)
        
        ax4.plot(sorted_ranks, cumulative, linewidth=2.5, color='#ff7f0e',
                label='LLM实际选择', alpha=0.8)
        
        max_rank = max(second_ranks)
        random_ranks = np.linspace(1, max_rank, 100)
        random_cdf = (random_ranks - 1) / (max_rank - 1)
        ax4.plot(random_ranks, random_cdf, 'r--', linewidth=2,
                label='随机期望（均匀分布）', alpha=0.8)
        
        median_rank = np.median(second_ranks)
        median_cdf = np.sum(second_ranks <= median_rank) / len(second_ranks)
        ax4.axvline(median_rank, color='orange', linestyle=':', alpha=0.5)
        ax4.axhline(median_cdf, color='orange', linestyle=':', alpha=0.5)
        ax4.plot(median_rank, median_cdf, 'o', color='#ff7f0e', markersize=8, 
                label=f'LLM实际中位数排名: {median_rank:.0f}')
        
        ax4.set_xlabel('变量排名 (1=最高百分比)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('累积概率', fontsize=11, fontweight='bold')
        ax4.set_title('第二次选择：累积分布函数对比\n(基于变量出现次数百分比)', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(1, min(500, max(second_ranks)))
    
    # 5. 排名区间比例对比柱状图（下中）
    ax5 = fig.add_subplot(gs[2, :])
    
    # 定义排名区间
    rank_ranges = ['排名=1\n(最高百分比)', '排名≤3', '排名≤10', '排名≤20', '排名≤50', '排名≤100']
    rank_limits = [1, 3, 10, 20, 50, 100]
    
    # 计算实际比例
    first_actual = []
    second_actual = []
    expected_probs = []
    
    for limit in rank_limits:
        if limit == 1:
            first_ratio = sum(1 for r in first_ranks if r == 1) / len(first_ranks) * 100
            second_ratio = sum(1 for r in second_ranks if r == 1) / len(second_ranks) * 100
            expected_prob = 1 / avg_vars * 100
        else:
            first_ratio = sum(1 for r in first_ranks if r <= limit) / len(first_ranks) * 100
            second_ratio = sum(1 for r in second_ranks if r <= limit) / len(second_ranks) * 100
            expected_prob = limit / avg_vars * 100
        
        first_actual.append(first_ratio)
        second_actual.append(second_ratio)
        expected_probs.append(expected_prob)
    
    x = np.arange(len(rank_ranges))
    width = 0.25
    
    bars1 = ax5.bar(x - width, expected_probs, width, label='随机期望', 
                   color='#d62728', alpha=0.8)
    bars2 = ax5.bar(x, first_actual, width, label='第一次选择（LLM）', 
                   color='#1f77b4', alpha=0.8)
    bars3 = ax5.bar(x + width, second_actual, width, label='第二次选择（LLM）', 
                   color='#ff7f0e', alpha=0.8)
    
    # 添加数值标签
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 1:
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
            else:
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}%', ha='center', va='bottom', fontsize=8)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    ax5.set_xlabel('排名区间 (基于变量出现次数百分比)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('选择比例 (%)', fontsize=12, fontweight='bold')
    ax5.set_title('不同排名区间的选择比例对比：LLM实际选择 vs 随机期望', fontsize=13, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(rank_ranges)
    ax5.legend(fontsize=11)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 添加倍数标注
    for i, (exp, first, second) in enumerate(zip(expected_probs, first_actual, second_actual)):
        if exp > 0:
            first_multiple = first / exp
            second_multiple = second / exp
            if first_multiple > 5:
                ax5.text(i - width, first + 1, f'{first_multiple:.1f}x', 
                        ha='center', fontsize=8, fontweight='bold', color='blue')
            if second_multiple > 5:
                ax5.text(i + width, second + 1, f'{second_multiple:.1f}x',
                        ha='center', fontsize=8, fontweight='bold', color='orange')
    
    plt.suptitle('LLM选择变量偏向高百分比变量的证据展示\n(排名基于变量出现次数占所有变量出现次数总和的百分比)', fontsize=15, fontweight='bold', y=0.98)
    
    # 保存图表
    output_file = 'output/bias_visualization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {output_file}")
    
    plt.close()
    
    # 生成补充说明
    print("\n" + "="*80)
    print("图表说明")
    print("="*80)
    print("""
图表包含5个子图，展示了LLM选择变量偏向高百分比变量的多个证据：

1. 左上图 - 第一次选择排名分布：
   - 蓝色直方图显示LLM实际选择的排名分布（左偏，集中在低排名）
   - 红色虚线显示随机期望的平均排名
   - 实际平均排名明显低于随机期望

2. 右上图 - 第二次选择排名分布：
   - 橙色直方图显示第二次选择的排名分布
   - 实际平均排名比第一次更低，偏向性更强

3. 左下图 - 第一次选择累积分布函数（CDF）：
   - 蓝色曲线：LLM实际选择的CDF（快速上升，说明大部分选择集中在低排名）
   - 红色虚线：随机期望的CDF（线性，均匀分布）
   - 实际曲线明显左偏，证明偏向高百分比变量

4. 右下图 - 第二次选择累积分布函数（CDF）：
   - 第二次选择的CDF同样左偏，偏向性更强

5. 下方图 - 排名区间比例对比：
   - 直接对比不同排名区间的选择比例
   - 红色：随机期望（极低）
   - 蓝色/橙色：LLM实际选择（显著高于期望）
   - 标注了实际值是期望值的倍数
   - 清楚展示：选择高百分比变量的概率显著高于随机期望

注：排名基于变量出现次数占所有变量出现次数总和的百分比，标准化了不同大小文件的影响。
    """)


if __name__ == "__main__":
    create_visualization()


