import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from loguru import logger


def _extract_fields(result_list):
    """Return (original_time, execution_time, status, solve_time_for_speedup).

    Supports both the legacy format and the current format produced by run_predictor.py.
    """
    if not isinstance(result_list, list) or len(result_list) < 4:
        return None

    original_time = result_list[1]
    execution_time = result_list[3]

    # Current format (run_predictor.py):
    # [orig_status, orig_time, orig_timeout,
    #  total_execution_time, total_solve_time, final_solve_time, llm_total_time,
    #  status, last_assignments, counterexamples_list]
    if len(result_list) >= 8 and isinstance(result_list[7], str):
        status = result_list[7]
        solve_time = result_list[5] if len(result_list) > 5 else None
        return original_time, execution_time, status, solve_time

    # Legacy format:
    # [orig_status, orig_time, orig_timeout, execution_time, status, solve_time, ...]
    status = result_list[4] if len(result_list) > 4 else 'unknown'
    solve_time = result_list[5] if len(result_list) > 5 else None
    return original_time, execution_time, status, solve_time


def analyze_results(result_file):
    logger.info(f'分析结果文件: {result_file}')

    with open(result_file, 'r') as f:
        results = json.load(f)

    stats = {
        'total': len(results),
        'succeed': 0,
        'failed': 0,
        'solve_times': [],
        'total_times': [],
        'original_times': [],
        'speedup_ratios': [],
    }

    time_bins = {
        '0-10s': 0,
        '10-60s': 0,
        '60-300s': 0,
        '300-600s': 0,
        '600-1200s': 0,
        '>1200s': 0,
    }

    for _, result_list in results.items():
        extracted = _extract_fields(result_list)
        if extracted is None:
            continue
        original_time, execution_time, status, solve_time = extracted

        stats['total_times'].append(execution_time)
        stats['original_times'].append(original_time)

        if status == 'succeed':
            stats['succeed'] += 1
            if solve_time is not None:
                stats['solve_times'].append(solve_time)

                if original_time and original_time > 0:
                    stats['speedup_ratios'].append(original_time / solve_time)

                if solve_time < 10:
                    time_bins['0-10s'] += 1
                elif solve_time < 60:
                    time_bins['10-60s'] += 1
                elif solve_time < 300:
                    time_bins['60-300s'] += 1
                elif solve_time < 600:
                    time_bins['300-600s'] += 1
                elif solve_time < 1200:
                    time_bins['600-1200s'] += 1
                else:
                    time_bins['>1200s'] += 1
        elif status == 'failed':
            stats['failed'] += 1

    stats['success_rate'] = stats['succeed'] / stats['total'] if stats['total'] > 0 else 0

    if stats['solve_times']:
        stats['avg_solve_time'] = float(np.mean(stats['solve_times']))
        stats['median_solve_time'] = float(np.median(stats['solve_times']))
        stats['min_solve_time'] = float(np.min(stats['solve_times']))
        stats['max_solve_time'] = float(np.max(stats['solve_times']))

    if stats['speedup_ratios']:
        stats['avg_speedup'] = float(np.mean(stats['speedup_ratios']))
        stats['median_speedup'] = float(np.median(stats['speedup_ratios']))

    stats['time_distribution'] = time_bins
    return stats


def print_analysis(stats):
    print('=' * 80)
    print('QF_NIA求解结果分析')
    print('=' * 80)
    print(f'总问题数: {stats["total"]}')
    print(f'成功求解: {stats["succeed"]} ({stats["success_rate"] * 100:.2f}%)')
    print(f'失败求解: {stats["failed"]} ({(1 - stats["success_rate"]) * 100:.2f}%)')
    print()

    if 'avg_solve_time' in stats:
        print('求解时间统计:')
        print(f'  平均时间: {stats["avg_solve_time"]:.2f}秒')
        print(f'  中位数时间: {stats["median_solve_time"]:.2f}秒')
        print(f'  最小时间: {stats["min_solve_time"]:.2f}秒')
        print(f'  最大时间: {stats["max_solve_time"]:.2f}秒')
        print()

    if 'avg_speedup' in stats:
        print('加速比统计:')
        print(f'  平均加速比: {stats["avg_speedup"]:.2f}x')
        print(f'  中位数加速比: {stats["median_speedup"]:.2f}x')
        print()

    if stats.get('time_distribution'):
        print('求解时间分布:')
        for time_range, count in stats['time_distribution'].items():
            percentage = count / stats['succeed'] * 100 if stats['succeed'] > 0 else 0
            print(f'  {time_range}: {count} ({percentage:.2f}%)')

    print('=' * 80)


def plot_time_distribution(stats, output_dir='supervenn_output'):
    os.makedirs(output_dir, exist_ok=True)

    if stats['solve_times']:
        plt.figure(figsize=(10, 6))
        plt.hist(stats['solve_times'], bins=50, edgecolor='black')
        plt.xlabel('求解时间 (秒)')
        plt.ylabel('问题数量')
        plt.title('QF_NIA求解时间分布')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'solve_time_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

    if stats['speedup_ratios']:
        plt.figure(figsize=(10, 6))
        plt.hist(stats['speedup_ratios'], bins=50, edgecolor='black')
        plt.xlabel('加速比')
        plt.ylabel('问题数量')
        plt.title('QF_NIA求解加速比分布')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'speedup_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

    if stats.get('time_distribution'):
        plt.figure(figsize=(12, 6))
        time_ranges = list(stats['time_distribution'].keys())
        counts = list(stats['time_distribution'].values())
        plt.bar(time_ranges, counts, edgecolor='black')
        plt.xlabel('时间范围')
        plt.ylabel('问题数量')
        plt.title('QF_NIA求解时间范围分布')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.savefig(os.path.join(output_dir, 'time_range_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()


def compare_solvers(result_files, solver_names):
    all_stats = {}
    for result_file, solver_name in zip(result_files, solver_names):
        if os.path.exists(result_file):
            all_stats[solver_name] = analyze_results(result_file)
        else:
            logger.warning(f'文件不存在: {result_file}')

    if not all_stats:
        logger.error('没有可分析的结果文件')
        return

    print('=' * 100)
    print('求解器性能比较')
    print('=' * 100)
    print(f'{"求解器":<15} {"总问题":<10} {"成功":<10} {"成功率":<12} {"平均时间":<12} {"平均加速比":<12}')
    print('-' * 100)

    for solver_name, stats in all_stats.items():
        success_rate = f'{stats["success_rate"] * 100:.2f}%'
        avg_time = f'{stats.get("avg_solve_time", 0):.2f}s'
        avg_speedup = f'{stats.get("avg_speedup", 0):.2f}x'
        print(
            f'{solver_name:<15} {stats["total"]:<10} {stats["succeed"]:<10} '
            f'{success_rate:<12} {avg_time:<12} {avg_speedup:<12}'
        )

    print('=' * 100)

    os.makedirs('supervenn_output', exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    solvers = list(all_stats.keys())

    success_rates = [all_stats[s]['success_rate'] * 100 for s in solvers]
    axes[0, 0].bar(solvers, success_rates, edgecolor='black')
    axes[0, 0].set_ylabel('成功率 (%)')
    axes[0, 0].set_title('求解器成功率比较')
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    avg_times = [all_stats[s].get('avg_solve_time', 0) for s in solvers]
    axes[0, 1].bar(solvers, avg_times, edgecolor='black', color='orange')
    axes[0, 1].set_ylabel('平均求解时间 (秒)')
    axes[0, 1].set_title('求解器平均求解时间比较')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    avg_speedups = [all_stats[s].get('avg_speedup', 0) for s in solvers]
    axes[1, 0].bar(solvers, avg_speedups, edgecolor='black', color='green')
    axes[1, 0].set_ylabel('平均加速比')
    axes[1, 0].set_title('求解器平均加速比比较')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    succeeds = [all_stats[s]['succeed'] for s in solvers]
    fails = [all_stats[s]['failed'] for s in solvers]
    x = np.arange(len(solvers))
    axes[1, 1].bar(x, succeeds, label='成功', edgecolor='black')
    axes[1, 1].bar(x, fails, bottom=succeeds, label='失败', edgecolor='black', color='red')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(solvers)
    axes[1, 1].set_ylabel('问题数量')
    axes[1, 1].set_title('求解器成功/失败分布')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('supervenn_output/solver_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_analysis_report(stats, output_file='solver_time_analysis.json'):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='分析QF_NIA求解结果')
    parser.add_argument('--result_file', type=str, required=True, help='结果JSON文件路径')
    parser.add_argument('--compare', nargs='+', help='要比较的结果文件列表')
    parser.add_argument('--names', nargs='+', help='求解器名称列表（用于比较）')
    parser.add_argument('--plot', action='store_true', help='是否生成图表')

    args = parser.parse_args()

    if args.compare and args.names:
        compare_solvers(args.compare, args.names)
    else:
        stats = analyze_results(args.result_file)
        print_analysis(stats)
        save_analysis_report(stats)
        if args.plot:
            plot_time_distribution(stats)
