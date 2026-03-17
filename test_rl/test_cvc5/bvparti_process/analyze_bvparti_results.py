#!/usr/bin/env python3
"""
BVParti结果分析器
分析BVParti求解器的批处理结果，生成详细的性能报告
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from loguru import logger

class BVPartiResultAnalyzer:
    """BVParti结果分析器"""
    
    def __init__(self, results_file):
        """
        初始化分析器
        
        Args:
            results_file: 结果JSON文件路径
        """
        self.results_file = Path(results_file)
        self.results = self.load_results()
        self.output_dir = self.results_file.parent / "analysis"
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Loaded {len(self.results)} results for analysis")
    
    def load_results(self):
        """加载结果文件"""
        try:
            with open(self.results_file, 'r') as f:
                results = json.load(f)
            return results
        except Exception as e:
            logger.error(f"Failed to load results from {self.results_file}: {e}")
            return {}
    
    def generate_basic_statistics(self):
        """生成基本统计信息"""
        if not self.results:
            logger.warning("No results to analyze")
            return {}
        
        stats = {
            'total_files': len(self.results),
            'sat': 0,
            'unsat': 0,
            'timeout': 0,
            'unknown': 0,
            'error': 0,
            'success_count': 0,
            'success_rate': 0,
            'avg_time': 0,
            'median_time': 0,
            'min_time': float('inf'),
            'max_time': 0,
            'total_time': 0,
            'std_time': 0
        }
        
        times = []
        success_times = []
        
        for filename, result in self.results.items():
            result_type = result.get('result', 'unknown')
            exec_time = result.get('execution_time', 0)
            
            # 统计结果类型
            if result_type in stats:
                stats[result_type] += 1
            
            # 统计时间
            if exec_time > 0:
                times.append(exec_time)
                stats['total_time'] += exec_time
                stats['min_time'] = min(stats['min_time'], exec_time)
                stats['max_time'] = max(stats['max_time'], exec_time)
                
                if result.get('success', False):
                    success_times.append(exec_time)
        
        # 计算统计量
        stats['success_count'] = stats['sat'] + stats['unsat']
        stats['success_rate'] = stats['success_count'] / stats['total_files'] * 100 if stats['total_files'] > 0 else 0
        
        if times:
            stats['avg_time'] = np.mean(times)
            stats['median_time'] = np.median(times)
            stats['std_time'] = np.std(times)
        
        if stats['min_time'] == float('inf'):
            stats['min_time'] = 0
        
        # 保存统计信息
        stats_file = self.output_dir / "basic_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info("Basic statistics generated:")
        logger.info(f"- Success rate: {stats['success_rate']:.1f}%")
        logger.info(f"- Average time: {stats['avg_time']:.2f}s")
        logger.info(f"- Total time: {stats['total_time']:.2f}s")
        
        return stats
    
    def generate_time_distribution_plot(self):
        """生成时间分布图"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, skipping plots")
            return
        
        times = []
        success_times = []
        timeout_times = []
        
        for result in self.results.values():
            exec_time = result.get('execution_time', 0)
            result_type = result.get('result', 'unknown')
            
            if exec_time > 0:
                times.append(exec_time)
                
                if result.get('success', False):
                    success_times.append(exec_time)
                elif result_type == 'timeout':
                    timeout_times.append(exec_time)
        
        if not times:
            logger.warning("No timing data available for plotting")
            return
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 时间分布直方图
        ax1.hist(times, bins=50, alpha=0.7, label='All', color='blue')
        if success_times:
            ax1.hist(success_times, bins=50, alpha=0.7, label='Success', color='green')
        if timeout_times:
            ax1.hist(timeout_times, bins=50, alpha=0.7, label='Timeout', color='red')
        
        ax1.set_xlabel('Execution Time (seconds)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Execution Time Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 累积分布图
        sorted_times = np.sort(times)
        y_vals = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
        ax2.plot(sorted_times, y_vals, label='All', linewidth=2)
        
        if success_times:
            sorted_success_times = np.sort(success_times)
            y_success = np.arange(1, len(sorted_success_times) + 1) / len(sorted_success_times)
            ax2.plot(sorted_success_times, y_success, label='Success', linewidth=2)
        
        ax2.set_xlabel('Execution Time (seconds)')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('Cumulative Distribution of Execution Times')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 保存图表
        plot_file = self.output_dir / "time_distribution.png"
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Time distribution plot saved to: {plot_file}")
    
    def generate_result_breakdown_plot(self):
        """生成结果分类饼图"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        
        # 统计各类结果
        result_counts = {}
        for result in self.results.values():
            result_type = result.get('result', 'unknown')
            result_counts[result_type] = result_counts.get(result_type, 0) + 1
        
        if not result_counts:
            return
        
        # 创建饼图
        fig, ax = plt.subplots(figsize=(10, 8))
        
        labels = list(result_counts.keys())
        sizes = list(result_counts.values())
        colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#34495e']
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors[:len(labels)], 
                                         autopct='%1.1f%%', startangle=90)
        
        ax.set_title('BVParti Solver Results Distribution', fontsize=16, fontweight='bold')
        
        # 添加图例
        ax.legend(wedges, [f'{label}: {size}' for label, size in zip(labels, sizes)],
                 title="Results", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        # 保存图表
        plot_file = self.output_dir / "result_breakdown.png"
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Result breakdown plot saved to: {plot_file}")
    
    def generate_detailed_report(self):
        """生成详细的文本报告"""
        report_lines = []
        report_lines.append("BVParti Solver Analysis Report")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Results file: {self.results_file}")
        report_lines.append("")
        
        # 基本统计
        stats = self.generate_basic_statistics()
        report_lines.append("Basic Statistics:")
        report_lines.append("-" * 20)
        report_lines.append(f"Total files processed: {stats['total_files']}")
        report_lines.append(f"SAT results: {stats['sat']}")
        report_lines.append(f"UNSAT results: {stats['unsat']}")
        report_lines.append(f"Timeout results: {stats['timeout']}")
        report_lines.append(f"Unknown results: {stats['unknown']}")
        report_lines.append(f"Error results: {stats['error']}")
        report_lines.append(f"Success rate: {stats['success_rate']:.2f}%")
        report_lines.append("")
        
        # 时间统计
        report_lines.append("Timing Statistics:")
        report_lines.append("-" * 20)
        report_lines.append(f"Average execution time: {stats['avg_time']:.2f}s")
        report_lines.append(f"Median execution time: {stats['median_time']:.2f}s")
        report_lines.append(f"Standard deviation: {stats['std_time']:.2f}s")
        report_lines.append(f"Minimum time: {stats['min_time']:.2f}s")
        report_lines.append(f"Maximum time: {stats['max_time']:.2f}s")
        report_lines.append(f"Total time: {stats['total_time']:.2f}s")
        report_lines.append("")
        
        # 错误分析
        error_files = []
        timeout_files = []
        for filename, result in self.results.items():
            if result.get('result') == 'error':
                error_files.append((filename, result.get('stderr', 'Unknown error')))
            elif result.get('result') == 'timeout':
                timeout_files.append(filename)
        
        if error_files:
            report_lines.append("Error Files:")
            report_lines.append("-" * 20)
            for filename, error in error_files[:10]:  # 只显示前10个
                report_lines.append(f"- {filename}: {error}")
            if len(error_files) > 10:
                report_lines.append(f"... and {len(error_files) - 10} more")
            report_lines.append("")
        
        if timeout_files:
            report_lines.append(f"Timeout Files ({len(timeout_files)}):")
            report_lines.append("-" * 20)
            for filename in timeout_files[:10]:  # 只显示前10个
                report_lines.append(f"- {filename}")
            if len(timeout_files) > 10:
                report_lines.append(f"... and {len(timeout_files) - 10} more")
            report_lines.append("")
        
        # 保存报告
        report_file = self.output_dir / "detailed_report.txt"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Detailed report saved to: {report_file}")
        
        # 也打印到控制台
        print('\n'.join(report_lines))
    
    def run_full_analysis(self):
        """运行完整分析"""
        logger.info("Starting full analysis...")
        
        # 生成各种分析
        self.generate_basic_statistics()
        self.generate_time_distribution_plot()
        self.generate_result_breakdown_plot()
        self.generate_detailed_report()
        
        logger.info(f"Analysis completed. Results saved to: {self.output_dir}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="BVParti Results Analyzer")
    parser.add_argument('results_file', help="Path to results JSON file")
    parser.add_argument('--output-dir', help="Output directory (default: same as results file)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        logger.error(f"Results file not found: {args.results_file}")
        return 1
    
    # 设置日志
    logger.add("logs/analyze_bvparti_{time}.log", rotation="100 MB", retention="7 days")
    
    try:
        analyzer = BVPartiResultAnalyzer(args.results_file)
        analyzer.run_full_analysis()
        return 0
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
