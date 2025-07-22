#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import os
import json
import glob
from collections import defaultdict

def read_json_log_file(log_file_path):
    """
    读取JSON格式的日志文件，提取text字段
    
    Args:
        log_file_path: 日志文件路径
        
    Returns:
        list: 包含所有text字段的列表
    """
    text_contents = []
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    # 解析JSON对象
                    json_obj = json.loads(line)
                    if "text" in json_obj:
                        text_contents.append(json_obj["text"])
                except json.JSONDecodeError:
                    # 如果不是有效的JSON，直接添加整行
                    text_contents.append(line)
    except Exception as e:
        print(f"读取文件 {log_file_path} 时出错: {str(e)}")
    
    return text_contents

def extract_file_path_from_log_texts(texts):
    """
    从日志文本中提取被处理的约束文件路径
    
    Args:
        texts: 日志文本列表
        
    Returns:
        str: 约束文件路径，如果未找到则返回None
    """
    file_pattern = r'开始处理文件 \d+/\d+: (.+)'
    subprocess_pattern = r'子进程开始处理: (.+)'
    
    for text in texts:
        file_match = re.search(file_pattern, text)
        if file_match:
            return file_match.group(1)
        
        subprocess_match = re.search(subprocess_pattern, text)
        if subprocess_match:
            return subprocess_match.group(1)
    
    return None

def extract_solve_times_from_log_texts(texts, file_path=None):
    """
    从日志文本列表中提取求解时间信息
    
    Args:
        texts: 日志文本列表
        file_path: 约束文件路径，如果为None则尝试从日志中提取
        
    Returns:
        dict: 包含各行求解时间的字典
    """
    result = {
        'line_190_times': [],  # handle_satisfiable 行的求解时间
        'line_196_times': [],  # handle_unknown 行的求解时间
        'line_203_times': [],  # handle_unsatisfiable 行的求解时间
        'line_302_times': [],  # 第302行的求解时间 (变量相关断言求解成功)
        'line_304_times': [],  # 第304行的求解时间 (变量相关断言求解失败: unsat/timeout)
        'total_calculated': 0,  # 根据日志计算的总求解时间
        'total_reported': 0,    # 日志中报告的最终累计求解时间
        'file_path': file_path  # 约束文件路径
    }
    
    # 正则表达式模式
    sat_pattern = r'求解结果: sat, 本次求解耗时: (\d+\.\d+)秒'
    unknown_pattern = r'求解结果: unknown, 本次求解耗时: (\d+\.\d+)秒'
    unsat_pattern = r'求解结果: unsat, 本次求解耗时: (\d+\.\d+)秒'
    cumulative_solve_time_pattern = r'当前累计求解时间: (\d+\.\d+)秒'
    final_result_pattern = r'约束已成功求解! 最终求解耗时: (\d+\.\d+)秒, 当前累计求解时间: (\d+\.\d+)秒'
    env_data_pattern = r'总求解时间=(\d+\.\d+)秒'
    file_pattern = r'开始处理文件 \d+/\d+: (.+)'
    subprocess_pattern = r'子进程开始处理: (.+)'
    
    # 新增: 第302行和304行的模式
    line_302_pattern = r'__mp_main__:step:302 - 变量 .+ 的相关断言求解成功：sat，耗时: (\d+\.\d+)秒'
    line_304_pattern = r'__mp_main__:step:304 - 变量 .+ 的相关断言求解失败: (unsat|timeout)，耗时: (\d+\.\d+)秒'
    
    # 使用集合跟踪已处理的日志行，避免重复处理
    processed_lines = set()
    
    # 如果没有提供文件路径，尝试从日志中提取
    if result['file_path'] is None:
        result['file_path'] = extract_file_path_from_log_texts(texts)
    
    # 提取所有求解时间信息
    for text in texts:
        # 提取sat求解时间 (line 190)
        sat_match = re.search(sat_pattern, text)
        if sat_match and text not in processed_lines:
            processed_lines.add(text)
            solve_time = float(sat_match.group(1))
            result['line_190_times'].append(solve_time)
            result['total_calculated'] += solve_time
        
        # 提取unknown求解时间 (line 196)
        unknown_match = re.search(unknown_pattern, text)
        if unknown_match and text not in processed_lines:
            processed_lines.add(text)
            solve_time = float(unknown_match.group(1))
            result['line_196_times'].append(solve_time)
            result['total_calculated'] += solve_time
        
        # 提取unsat求解时间 (line 203)
        unsat_match = re.search(unsat_pattern, text)
        if unsat_match and text not in processed_lines:
            processed_lines.add(text)
            solve_time = float(unsat_match.group(1))
            result['line_203_times'].append(solve_time)
            result['total_calculated'] += solve_time
        
        # 新增: 提取第302行的求解时间 (变量相关断言求解成功)
        line_302_match = re.search(line_302_pattern, text)
        if line_302_match and text not in processed_lines:
            processed_lines.add(text)
            solve_time = float(line_302_match.group(1))
            result['line_302_times'].append(solve_time)
            result['total_calculated'] += solve_time
        
        # 新增: 提取第304行的求解时间 (变量相关断言求解失败: unsat或timeout)
        line_304_match = re.search(line_304_pattern, text)
        if line_304_match and text not in processed_lines:
            processed_lines.add(text)
            solve_time = float(line_304_match.group(2))  # 注意：组索引从1变为2，因为现在有两个捕获组
            result['line_304_times'].append(solve_time)
            result['total_calculated'] += solve_time
        
        # 提取最终累计求解时间
        final_match = re.search(final_result_pattern, text)
        if final_match:
            total_cumulative_time = float(final_match.group(2))
            result['total_reported'] = total_cumulative_time
        
        # 提取环境数据更新中的总求解时间
        env_data_match = re.search(env_data_pattern, text)
        if env_data_match and result['total_reported'] == 0:
            total_solve_time = float(env_data_match.group(1))
            result['total_reported'] = total_solve_time
    
    # 如果没有找到最终求解时间，则查找最后一次记录的累计求解时间
    if result['total_reported'] == 0:
        for text in reversed(texts):
            cumulative_match = re.search(cumulative_solve_time_pattern, text)
            if cumulative_match:
                result['total_reported'] = float(cumulative_match.group(1))
                break
    
    return result

def extract_file_processing_segments(texts):
    """
    从日志文本列表中提取"子进程开始处理"到"完成文件"之间的约束处理段
    
    Args:
        texts: 日志文本列表
        
    Returns:
        dict: 文件路径到处理段的映射
    """
    segments = {}
    current_file = None
    current_segment = []
    
    # 正则表达式模式
    start_pattern = r'子进程开始处理: (.+)'
    end_pattern = r'完成文件 \d+/\d+: (.+)'
    
    for text in texts:
        # 检查是否是新文件的开始
        start_match = re.search(start_pattern, text)
        if start_match:
            # 如果有当前处理的文件，保存之前的段
            if current_file and current_segment:
                segments[current_file] = current_segment
            
            # 开始新文件的处理
            current_file = start_match.group(1)
            current_segment = [text]
            continue
        
        # 检查是否是文件处理的结束
        end_match = re.search(end_pattern, text)
        if end_match and current_file == end_match.group(1):
            # 添加结束行
            current_segment.append(text)
            # 保存当前段
            segments[current_file] = current_segment
            # 重置
            current_file = None
            current_segment = []
            continue
        
        # 如果正在处理文件，添加到当前段
        if current_file:
            current_segment.append(text)
    
    # 处理最后一个段
    if current_file and current_segment:
        segments[current_file] = current_segment
    
    return segments

def collect_all_log_files(log_dir):
    """
    收集日志目录下的所有日志文件
    
    Args:
        log_dir: 日志目录路径
        
    Returns:
        list: 日志文件路径列表
    """
    log_files = []
    for root, _, files in os.walk(log_dir):
        for file in files:
            if file.endswith('.log'):
                log_path = os.path.join(root, file)
                log_files.append(log_path)
    return log_files

def process_all_log_files(log_dir):
    """
    处理所有日志文件，提取文件处理段
    
    Args:
        log_dir: 日志目录路径
        
    Returns:
        dict: 文件路径到处理段的映射
    """
    all_segments = {}
    log_files = collect_all_log_files(log_dir)
    
    print(f"发现 {len(log_files)} 个日志文件")
    
    for log_file in log_files:
        print(f"处理日志文件: {os.path.basename(log_file)}")
        texts = read_json_log_file(log_file)
        segments = extract_file_processing_segments(texts)
        
        # 合并段
        for file_path, segment in segments.items():
            if file_path in all_segments:
                all_segments[file_path].extend(segment)
            else:
                all_segments[file_path] = segment
    
    print(f"总共提取了 {len(all_segments)} 个文件的处理段")
    return all_segments

def analyze_info_dict_vs_logs(info_dict_path, log_dir, output_file=None, update_info_dict=True):
    """
    分析info_dict中的求解时间与日志中的求解时间的差异
    
    Args:
        info_dict_path: info_dict文件路径
        log_dir: 日志目录路径
        output_file: 输出文件路径，如果为None则只打印结果
        update_info_dict: 是否更新info_dict中的求解时间
        
    Returns:
        dict: 分析结果
    """
    # 加载info_dict
    with open(info_dict_path, 'r', encoding='utf-8') as f:
        info_dict = json.load(f)
    
    # 初始化结果
    results = []
    stats = {
        'total_files': len(info_dict),
        'files_analyzed': 0,
        'files_with_diff': 0,
        'files_not_found': 0,
        'total_diff': 0,
        'max_diff': 0,
        'max_diff_file': None
    }
    
    # 处理所有日志文件，提取文件处理段
    print(f"正在处理日志目录: {log_dir}")
    all_segments = process_all_log_files(log_dir)
    
    # 是否有更新
    has_updates = False
    
    # 分析每个约束文件
    for constraint_path, info in info_dict.items():
        # 查找对应的处理段
        if constraint_path in all_segments:
            # 提取求解时间
            segment_texts = all_segments[constraint_path]
            solve_times = extract_solve_times_from_log_texts(segment_texts, constraint_path)
            
            # 确保info_dict中有足够的元素
            if len(info) >= 5:
                info_dict_time = info[4]  # 总求解时间索引
                log_total_time = solve_times['total_reported']
                log_calculated_time = solve_times['total_calculated']
                
                # 计算差异
                diff = abs(info_dict_time - log_total_time)
                
                # 如果需要更新info_dict
                if update_info_dict and log_calculated_time > 0:
                    # 更新info_dict中的求解时间
                    old_time = info[4]
                    info[4] = log_calculated_time
                    has_updates = True
                    print(f"更新 {constraint_path} 的求解时间: {old_time:.3f}秒 -> {log_calculated_time:.3f}秒")
                
                # 记录结果
                result = {
                    'constraint_path': constraint_path,
                    'info_dict_time': info_dict_time,
                    'log_reported_time': log_total_time,
                    'log_calculated_time': log_calculated_time,
                    'difference': diff,
                    'line_190_times': solve_times['line_190_times'],
                    'line_196_times': solve_times['line_196_times'],
                    'line_203_times': solve_times['line_203_times'],
                    'line_302_times': solve_times['line_302_times'],
                    'line_304_times': solve_times['line_304_times'],
                    'line_190_sum': sum(solve_times['line_190_times']),
                    'line_196_sum': sum(solve_times['line_196_times']),
                    'line_203_sum': sum(solve_times['line_203_times']),
                    'line_302_sum': sum(solve_times['line_302_times']),
                    'line_304_sum': sum(solve_times['line_304_times'])
                }
                
                results.append(result)
                stats['files_analyzed'] += 1
                
                # 更新统计信息
                if diff > 0.1:  # 差异大于0.1秒视为有差异
                    stats['files_with_diff'] += 1
                    stats['total_diff'] += diff
                    
                    if diff > stats['max_diff']:
                        stats['max_diff'] = diff
                        stats['max_diff_file'] = constraint_path
            else:
                print(f"警告: 文件 {constraint_path} 的信息条目格式不正确")
        else:
            stats['files_not_found'] += 1
            print(f"未找到约束 {constraint_path} 对应的日志文件")
    
    # 如果有更新，保存更新后的info_dict
    if has_updates and update_info_dict:
        updated_info_dict_path = info_dict_path.replace('.txt', '_updated.txt')
        with open(updated_info_dict_path, 'w', encoding='utf-8') as f:
            json.dump(info_dict, f, indent=2)
        print(f"已将更新后的info_dict保存到: {updated_info_dict_path}")
    
    # 构建完整结果
    analysis_result = {
        'stats': stats,
        'details': results
    }
    
    # 保存结果到文件
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, indent=2)
        print(f"分析结果已保存到: {output_file}")
    
    # 打印统计信息
    print("\n=== 分析统计 ===")
    print(f"总约束文件数: {stats['total_files']}")
    print(f"成功分析的文件数: {stats['files_analyzed']}")
    print(f"有时间差异的文件数: {stats['files_with_diff']}")
    print(f"未找到日志的文件数: {stats['files_not_found']}")
    print(f"总时间差异: {stats['total_diff']:.2f}秒")
    print(f"最大时间差异: {stats['max_diff']:.2f}秒 (文件: {stats['max_diff_file']})")
    
    # 打印详细的差异信息
    print("\n=== 详细差异 ===")
    for result in sorted(results, key=lambda x: x['difference'], reverse=True)[:10]:  # 打印差异最大的10个
        print(f"约束: {result['constraint_path']}")
        print(f"  info_dict时间: {result['info_dict_time']:.3f}秒")
        print(f"  日志报告时间: {result['log_reported_time']:.3f}秒")
        print(f"  日志计算时间: {result['log_calculated_time']:.3f}秒")
        print(f"  差异: {result['difference']:.3f}秒")
        print(f"  line_190 (sat) 次数: {len(result['line_190_times'])}, 总时间: {result['line_190_sum']:.3f}秒")
        print(f"  line_196 (unknown) 次数: {len(result['line_196_times'])}, 总时间: {result['line_196_sum']:.3f}秒")
        print(f"  line_203 (unsat) 次数: {len(result['line_203_times'])}, 总时间: {result['line_203_sum']:.3f}秒")
        print(f"  line_302 (变量相关断言求解成功) 次数: {len(result['line_302_times'])}, 总时间: {result['line_302_sum']:.3f}秒")
        print(f"  line_304 (变量相关断言求解失败: unsat/timeout) 次数: {len(result['line_304_times'])}, 总时间: {result['line_304_sum']:.3f}秒")
        print()
    
    return analysis_result

def main():
    """
    主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='分析info_dict与日志中的求解时间差异')
    parser.add_argument('info_dict_path', nargs='?', 
                        help='info_dict文件路径',
                        default='/home/nju/PycharmProjects/Pearl/test_rl/test_cvc5/mathsat5_process/info_dict_SMTimer_llama3.1:70b_1200s_info_dict_rl_mathsat5_0628.txt')
    parser.add_argument('log_dir', nargs='?',
                        help='日志目录路径',
                        default='/home/nju/PycharmProjects/Pearl/test_rl/test_cvc5/mathsat5_process/log/run_2025-06-27_23-37-26')
    parser.add_argument('--output', '-o', 
                        help='输出文件路径',
                        default='/home/nju/PycharmProjects/Pearl/test_rl/test_cvc5/mathsat5_process/solver_time_analysis.json')
    parser.add_argument('--update', '-u', action='store_true',
                        help='是否更新info_dict中的求解时间')
    parser.add_argument('--no-update', dest='update', action='store_false',
                        help='不更新info_dict中的求解时间')
    parser.set_defaults(update=True)
    
    args = parser.parse_args()
    
    analyze_info_dict_vs_logs(args.info_dict_path, args.log_dir, args.output, args.update)

if __name__ == "__main__":
    main() 