#!/usr/bin/env python3
"""
生成Z3的SuperVenn图
"""

import os
import sys
import json
sys.path.append('/home/lz/PycharmProjects/Pearl')

import matplotlib.pyplot as plt
from supervenn import supervenn

def plot_supervenn_diagrams_z3(baseline_solved, rl_llm_solved, solver_name="Z3", output_dir=None):
    """
    绘制Z3的SuperVenn图，比较baseline求解器和RL+LLM增强版本的求解能力
    
    Args:
        baseline_solved: baseline求解器解决的约束集合
        rl_llm_solved: RL+LLM增强版本解决的约束集合
        solver_name: 求解器名称，用于标签
        output_dir: 输出目录，如果为None则不保存文件
    """
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    set_labels = [f'{solver_name} Baseline', f'{solver_name} + RL+LLM']
    
    def plot_supervenn(sets_data, title, filename):
        # Professional color palette
        color_palette = ['#0072B2', '#D55E00']
        
        try:
            with plt.style.context('seaborn-v0_8-paper'):
                plt.rcParams.update({
                    'font.family': 'serif', 
                    'font.serif': ['Times New Roman', 'DejaVu Serif'],
                    'font.size': 10
                })
                plt.figure(figsize=(10, 3))
                plot = supervenn(sets_data, 
                          set_annotations=set_labels, 
                          side_plots=True,
                          sets_ordering='minimize gaps',
                          bar_height=1.0,
                          side_plot_width=0.4,
                          color_cycle=color_palette,
                          rotate_col_annotations=False,
                          widths_minmax_ratio=0.02)
                
                if 'main' in plot.axes:
                    plot.axes['main'].set_title(title, fontsize=14, fontweight='bold')
                
                if output_dir:
                    output_path = os.path.join(output_dir, filename)
                    plt.savefig(output_path, bbox_inches='tight', format='pdf', dpi=300)
                    print(f"SuperVenn diagram saved to: {output_path}")
                else:
                    plt.show()
                plt.close()
        except Exception as e:
            print(f"Error creating SuperVenn diagram: {e}")
            plt.close()
    
    # 创建求解能力比较图
    solved_sets = [baseline_solved, rl_llm_solved]
    plot_supervenn(solved_sets, f"{solver_name} Solving Capability Comparison", f"supervenn_{solver_name.lower()}_comparison.pdf")
    
    # 打印统计信息
    baseline_only = baseline_solved - rl_llm_solved
    rl_llm_only = rl_llm_solved - baseline_solved
    both_solved = baseline_solved & rl_llm_solved
    
    print(f"\n{solver_name} SuperVenn Analysis:")
    print(f"  {solver_name} Baseline only solved: {len(baseline_only)} constraints")
    print(f"  {solver_name} + RL+LLM only solved: {len(rl_llm_only)} constraints")
    print(f"  Both methods solved: {len(both_solved)} constraints")
    print(f"  Total unique constraints solved: {len(baseline_solved | rl_llm_solved)} constraints")
    
    return {
        'baseline_only': baseline_only,
        'rl_llm_only': rl_llm_only,
        'both_solved': both_solved,
        'baseline_total': len(baseline_solved),
        'rl_llm_total': len(rl_llm_solved),
        'union_total': len(baseline_solved | rl_llm_solved)
    }

def load_dictionary(file_path):
    """加载字典文件"""
    with open(file_path, 'r') as f:
        return json.load(f)

def generate_z3_supervenn():
    """生成Z3的SuperVenn图"""
    
    # Z3数据文件路径
    z3_baseline_path = '/home/lz/PycharmProjects/Pearl/test_rl/test_solve/info_dict_bingxing.txt'
    z3_rl_llm_path = '/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1110_pre_SMTimer_llama3.1:70b_1200s_info_dict_rl.txt'
    var_count_path = '/home/lz/PycharmProjects/Pearl/test_rl/test_solve/var_count.txt'
    output_dir = '/home/lz/PycharmProjects/Pearl/paper/pics'
    
    # 检查文件是否存在
    for path in [z3_baseline_path, z3_rl_llm_path, var_count_path]:
        if not os.path.exists(path):
            print(f"错误: 找不到文件 {path}")
            return False
    
    print("加载Z3数据...")
    z3_baseline = load_dictionary(z3_baseline_path)
    z3_rl_llm = load_dictionary(z3_rl_llm_path)
    var_count = load_dictionary(var_count_path)
    
    print(f"Z3 baseline数据: {len(z3_baseline)} 个约束")
    print(f"Z3 RL+LLM数据: {len(z3_rl_llm)} 个约束")
    print(f"变量统计数据: {len(var_count)} 个约束")
    
    # 构建求解集合（筛选条件：时间>300s，变量>5）
    z3_baseline_solved = set()
    z3_rl_llm_solved = set()
    
    timeout = 1200
    
    for k, v in z3_baseline.items():
        if k in var_count and len(var_count[k]) > 5 and v[1] > 300:
            # 处理超时
            if v[1] > timeout:
                v[0] = 'unknown'
                v[1] = timeout
            
            # Z3 baseline解决的约束
            if v[0] in ['sat', 'unsat']:
                z3_baseline_solved.add(k)
    
    for k, v in z3_rl_llm.items():
        if k in var_count and len(var_count[k]) > 5 and v[1] > 300:
            # 处理超时
            if v[3] > timeout:
                v[4] = 'failed'
                v[3] = timeout
            
            # RL+LLM解决的约束
            if v[4] == 'succeed':
                z3_rl_llm_solved.add(k)
    
    print(f"Z3 baseline解决: {len(z3_baseline_solved)} 个约束")
    print(f"Z3 RL+LLM解决: {len(z3_rl_llm_solved)} 个约束")
    
    # 生成SuperVenn图
    supervenn_stats = plot_supervenn_diagrams_z3(
        z3_baseline_solved, 
        z3_rl_llm_solved, 
        solver_name="Z3", 
        output_dir=output_dir
    )
    
    return supervenn_stats

if __name__ == "__main__":
    print("生成Z3 SuperVenn图...")
    print("=" * 60)
    
    stats = generate_z3_supervenn()
    
    if stats:
        print("\n" + "=" * 60)
        print("Z3 SuperVenn图生成完成!")
        print(f"输出位置: /home/lz/PycharmProjects/Pearl/paper/pics/supervenn_z3_comparison.pdf")
    else:
        print("Z3 SuperVenn图生成失败!")
