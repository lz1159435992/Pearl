"""
分析第二次选择变量的详细情况
1. 第二次选择的平均百分比
2. 第一次和第二次选择的重复性分析
"""
import json
import os
import numpy as np
from collections import defaultdict, Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')


def load_results():
    """加载实验结果"""
    results_file = os.path.join(OUTPUT_DIR, 'experiment_results.json')
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_second_selection(results):
    """分析第二次选择"""
    
    # 统计数据
    total_records = len(results)
    first_valid = 0
    second_valid = 0
    both_valid = 0
    
    # 重复性分析
    same_selection = 0  # 第一次和第二次选择相同变量
    different_selection = 0  # 第一次和第二次选择不同变量
    
    # 百分比数据
    second_percentages = []
    first_percentages = []
    
    # 按文件分组分析
    file_analysis = defaultdict(list)
    
    for result in results:
        all_var_counts = result.get('all_original_var_counts', {})
        if not all_var_counts:
            continue
        
        total_count = sum(all_var_counts.values())
        if total_count == 0:
            continue
        
        # 获取选择结果
        first_var = result.get('first_selected_original_var')
        second_var = result.get('second_selected_original_var')
        
        if first_var:
            first_valid += 1
            if first_var in all_var_counts:
                first_pct = (all_var_counts[first_var] / total_count) * 100
                first_percentages.append(first_pct)
        
        if second_var:
            second_valid += 1
            if second_var in all_var_counts:
                second_pct = (all_var_counts[second_var] / total_count) * 100
                second_percentages.append(second_pct)
        
        # 重复性分析
        if first_var and second_var:
            both_valid += 1
            if first_var == second_var:
                same_selection += 1
            else:
                different_selection += 1
            
            # 记录文件级别的分析
            file_idx = result.get('file_index')
            file_analysis[file_idx].append({
                'first': first_var,
                'second': second_var,
                'same': first_var == second_var
            })
    
    return {
        'total_records': total_records,
        'first_valid': first_valid,
        'second_valid': second_valid,
        'both_valid': both_valid,
        'same_selection': same_selection,
        'different_selection': different_selection,
        'first_percentages': first_percentages,
        'second_percentages': second_percentages,
        'file_analysis': file_analysis
    }


def print_report(data):
    """打印分析报告"""
    print("=" * 80)
    print("第二次选择变量分析报告")
    print("=" * 80)
    print()
    
    # 基本统计
    print("## 1. 基本统计")
    print("-" * 60)
    print(f"总记录数: {data['total_records']}")
    print(f"第一次选择有效: {data['first_valid']} ({data['first_valid']/data['total_records']*100:.1f}%)")
    print(f"第二次选择有效: {data['second_valid']} ({data['second_valid']/data['total_records']*100:.1f}%)")
    print(f"两次都有效: {data['both_valid']} ({data['both_valid']/data['total_records']*100:.1f}%)")
    print()
    
    # 第二次选择的百分比统计
    print("## 2. 第二次选择的平均百分比")
    print("-" * 60)
    if data['second_percentages']:
        second_pcts = np.array(data['second_percentages'])
        print(f"样本数: {len(second_pcts)}")
        print(f"平均百分比: {np.mean(second_pcts):.4f}%")
        print(f"标准差: {np.std(second_pcts):.4f}%")
        print(f"中位数: {np.median(second_pcts):.4f}%")
        print(f"最小值: {np.min(second_pcts):.4f}%")
        print(f"最大值: {np.max(second_pcts):.4f}%")
    print()
    
    # 与第一次选择对比
    print("## 3. 第一次 vs 第二次选择对比")
    print("-" * 60)
    if data['first_percentages'] and data['second_percentages']:
        first_pcts = np.array(data['first_percentages'])
        second_pcts = np.array(data['second_percentages'])
        print(f"第一次选择平均百分比: {np.mean(first_pcts):.4f}%")
        print(f"第二次选择平均百分比: {np.mean(second_pcts):.4f}%")
        print(f"差异: {np.mean(first_pcts) - np.mean(second_pcts):.4f}%")
        print(f"第二次/第一次: {np.mean(second_pcts)/np.mean(first_pcts)*100:.1f}%")
    print()
    
    # 重复性分析
    print("## 4. 重复性分析（第一次和第二次是否选择相同变量）")
    print("-" * 60)
    print(f"两次都有效的记录数: {data['both_valid']}")
    print(f"选择相同变量: {data['same_selection']} ({data['same_selection']/data['both_valid']*100:.2f}%)")
    print(f"选择不同变量: {data['different_selection']} ({data['different_selection']/data['both_valid']*100:.2f}%)")
    print()
    
    # 按文件分析重复性
    print("## 5. 按文件分析重复性")
    print("-" * 60)
    
    file_same_rates = []
    for file_idx, selections in data['file_analysis'].items():
        same_count = sum(1 for s in selections if s['same'])
        total = len(selections)
        rate = same_count / total * 100 if total > 0 else 0
        file_same_rates.append(rate)
    
    if file_same_rates:
        print(f"文件数: {len(file_same_rates)}")
        print(f"平均重复率: {np.mean(file_same_rates):.2f}%")
        print(f"重复率标准差: {np.std(file_same_rates):.2f}%")
        print(f"重复率中位数: {np.median(file_same_rates):.2f}%")
        print(f"重复率最小值: {np.min(file_same_rates):.2f}%")
        print(f"重复率最大值: {np.max(file_same_rates):.2f}%")
        
        # 重复率分布
        print()
        print("重复率分布:")
        bins = [0, 1, 5, 10, 20, 50, 100]
        for i in range(len(bins)-1):
            count = sum(1 for r in file_same_rates if bins[i] <= r < bins[i+1])
            print(f"  {bins[i]}%-{bins[i+1]}%: {count}个文件 ({count/len(file_same_rates)*100:.1f}%)")
        count_100 = sum(1 for r in file_same_rates if r == 100)
        print(f"  100%: {count_100}个文件 ({count_100/len(file_same_rates)*100:.1f}%)")
    
    print()
    print("=" * 80)
    print("## 结论")
    print("=" * 80)
    print()
    
    if data['both_valid'] > 0:
        repeat_rate = data['same_selection'] / data['both_valid'] * 100
        print(f"1. 第二次选择的平均百分比: {np.mean(data['second_percentages']):.4f}%")
        print(f"2. 第一次和第二次选择相同变量的比例: {repeat_rate:.2f}%")
        print()
        
        if repeat_rate < 5:
            print("结论: 重复率很低，LLM在第二次选择时几乎不会选择与第一次相同的变量。")
            print("这符合预期，因为prompt中明确要求选择'下一个'变量。")
        elif repeat_rate < 20:
            print("结论: 重复率较低，但存在一定比例的重复选择。")
            print("可能原因: LLM有时会忽略prompt中的约束，或者某些变量确实非常重要。")
        else:
            print("结论: 重复率较高，LLM经常选择与第一次相同的变量。")
            print("这可能表明LLM对某些变量有强烈偏好，或者prompt设计需要改进。")


def main():
    print("加载实验结果...")
    results = load_results()
    print(f"共加载 {len(results)} 条记录")
    print()
    
    print("分析第二次选择...")
    data = analyze_second_selection(results)
    
    print()
    print_report(data)


if __name__ == '__main__':
    main()
