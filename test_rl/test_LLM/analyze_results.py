"""
实验结果分析脚本
提供更详细的数据分析和可视化功能
"""
import json
import os
import sys
from collections import defaultdict, Counter
from pathlib import Path

def load_results(results_file):
    """加载实验结果"""
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_by_variable_position(results):
    """
    分析变量在映射中的位置（第1、2、3个）对选择的影响
    """
    # 统计每个位置的变量被选择的次数
    position_selections = defaultdict(int)  # {映射位置: 选择次数}
    
    for result in results:
        if not result.get('llm_selected_var') or not result.get('mapping'):
            continue
        
        selected_var = result['llm_selected_var']
        mapping = result['mapping']
        
        # 找出选择的变量在映射中的位置（原始变量列表中的位置）
        original_vars = result['original_variables']
        
        # 反向查找：VAR_i -> 原始变量
        for orig_var, var_name in mapping.items():
            if var_name == selected_var:
                # 找到原始变量在列表中的位置（0, 1, 2）
                position = original_vars.index(orig_var)
                position_selections[position] += 1
                break
    
    return dict(position_selections)


def analyze_by_occurrence_rank(results):
    """
    分析变量出现次数排名对选择的影响
    """
    # 统计选择第1/2/3高频变量的次数
    rank_selections = defaultdict(int)  # {排名: 选择次数}
    
    for result in results:
        if not result.get('llm_selected_var') or not result.get('original_var_counts'):
            continue
        
        # 按出现次数排序原始变量
        var_counts = result['original_var_counts']
        sorted_vars = sorted(var_counts.items(), key=lambda x: x[1], reverse=True)
        
        # 找出选择的原始变量
        selected_orig_var = result.get('selected_original_var')
        if not selected_orig_var:
            continue
        
        # 找出该变量在排序中的排名（0-based）
        for rank, (var, count) in enumerate(sorted_vars):
            if var == selected_orig_var:
                rank_selections[rank] += 1
                break
    
    return dict(rank_selections)


def analyze_consistency_across_mappings(results):
    """
    分析同一文件的不同映射组合中，LLM选择的一致性
    """
    # 按文件分组
    files_dict = defaultdict(list)
    for result in results:
        file_idx = result['file_index']
        files_dict[file_idx].append(result)
    
    consistency_stats = {
        'total_files': len(files_dict),
        'consistent_files': 0,
        'inconsistent_files': 0,
        'details': []
    }
    
    for file_idx, file_results in files_dict.items():
        # 收集每个映射组合选择的原始变量
        selected_vars = []
        for result in file_results:
            if result.get('selected_original_var'):
                selected_vars.append(result['selected_original_var'])
        
        if len(selected_vars) > 0:
            unique_selections = set(selected_vars)
            is_consistent = len(unique_selections) == 1
            
            if is_consistent:
                consistency_stats['consistent_files'] += 1
            else:
                consistency_stats['inconsistent_files'] += 1
            
            consistency_stats['details'].append({
                'file_index': file_idx,
                'total_mappings': len(file_results),
                'selected_count': len(selected_vars),
                'unique_selections': list(unique_selections),
                'is_consistent': is_consistent,
                'selected_vars': selected_vars
            })
    
    return consistency_stats


def generate_detailed_analysis(results_file, output_file):
    """生成详细分析报告"""
    results = load_results(results_file)
    
    analysis_lines = []
    analysis_lines.append("="*80)
    analysis_lines.append("详细数据分析报告")
    analysis_lines.append("="*80)
    analysis_lines.append("")
    
    # 1. 位置分析
    analysis_lines.append("="*80)
    analysis_lines.append("1. 变量在映射中的位置对选择的影响")
    analysis_lines.append("="*80)
    position_analysis = analyze_by_variable_position(results)
    
    if position_analysis:
        total = sum(position_analysis.values())
        analysis_lines.append("各位置被选择的次数:")
        for pos in sorted(position_analysis.keys()):
            count = position_analysis[pos]
            percentage = count / total * 100 if total > 0 else 0
            position_name = ['第1个变量', '第2个变量', '第3个变量'][pos]
            analysis_lines.append(f"  {position_name} (位置{pos}): {count}次 ({percentage:.1f}%)")
    else:
        analysis_lines.append("无有效数据")
    analysis_lines.append("")
    
    # 2. 出现次数排名分析
    analysis_lines.append("="*80)
    analysis_lines.append("2. 变量出现次数排名对选择的影响")
    analysis_lines.append("="*80)
    rank_analysis = analyze_by_occurrence_rank(results)
    
    if rank_analysis:
        total = sum(rank_analysis.values())
        analysis_lines.append("各排名被选择的次数:")
        rank_names = ['最多', '次多', '最少']
        for rank in sorted(rank_analysis.keys()):
            count = rank_analysis[rank]
            percentage = count / total * 100 if total > 0 else 0
            rank_name = rank_names[rank] if rank < len(rank_names) else f"排名{rank+1}"
            analysis_lines.append(f"  出现次数{rank_name}的变量: {count}次 ({percentage:.1f}%)")
    else:
        analysis_lines.append("无有效数据")
    analysis_lines.append("")
    
    # 3. 一致性分析
    analysis_lines.append("="*80)
    analysis_lines.append("3. 不同映射组合的选择一致性分析")
    analysis_lines.append("="*80)
    consistency_stats = analyze_consistency_across_mappings(results)
    
    analysis_lines.append(f"总文件数: {consistency_stats['total_files']}")
    analysis_lines.append(f"一致的文件数: {consistency_stats['consistent_files']}")
    analysis_lines.append(f"不一致的文件数: {consistency_stats['inconsistent_files']}")
    
    if consistency_stats['total_files'] > 0:
        consistency_rate = consistency_stats['consistent_files'] / consistency_stats['total_files'] * 100
        analysis_lines.append(f"一致性比例: {consistency_rate:.1f}%")
    
    analysis_lines.append("")
    analysis_lines.append("详细情况:")
    for detail in consistency_stats['details']:
        analysis_lines.append(f"  文件 {detail['file_index']}:")
        analysis_lines.append(f"    一致性: {'是' if detail['is_consistent'] else '否'}")
        analysis_lines.append(f"    唯一选择: {detail['unique_selections']}")
        analysis_lines.append(f"    所有选择: {detail['selected_vars']}")
    analysis_lines.append("")
    
    # 4. VAR_i选择偏好分析
    analysis_lines.append("="*80)
    analysis_lines.append("4. VAR_i选择偏好分析")
    analysis_lines.append("="*80)
    var_selection_counts = Counter()
    for result in results:
        if result.get('llm_selected_var'):
            var_selection_counts[result['llm_selected_var']] += 1
    
    total_selections = sum(var_selection_counts.values())
    analysis_lines.append("各VAR_i被选择的次数:")
    for var in ['VAR_1', 'VAR_2', 'VAR_3']:
        count = var_selection_counts.get(var, 0)
        percentage = count / total_selections * 100 if total_selections > 0 else 0
        analysis_lines.append(f"  {var}: {count}次 ({percentage:.1f}%)")
    analysis_lines.append("")
    
    # 保存报告
    report_text = '\n'.join(analysis_lines)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n详细分析报告已保存到: {output_file}")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_file = os.path.join(script_dir, 'output', 'experiment_results.json')
    output_file = os.path.join(script_dir, 'output', 'detailed_analysis.txt')
    
    if not os.path.exists(results_file):
        print(f"错误: 找不到结果文件 {results_file}")
        print("请先运行 experiment.py 生成实验结果")
        sys.exit(1)
    
    generate_detailed_analysis(results_file, output_file)

