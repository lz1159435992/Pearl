#!/usr/bin/env python3
"""
BVParti批处理求解器
用于批量处理SMT2文件，类似于AriParti项目中的batch_process.py
"""

import os
import sys
import json
import time
import glob
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from loguru import logger

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_bvparti_predictor import BVPartiSolver, BVPartiConfig

class BVPartiBatchProcessor:
    """BVParti批处理器"""
    
    def __init__(self, 
                 output_dir="batch_output",
                 max_workers=4,
                 time_limit=300,
                 result_filename="bvparti_results.json"):
        """
        初始化批处理器
        
        Args:
            output_dir: 输出目录
            max_workers: 最大并行工作进程数
            time_limit: 每个实例的时间限制（秒）
            result_filename: 结果文件名
        """
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        self.time_limit = time_limit
        self.result_file = self.output_dir / result_filename
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建日志目录
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载已有结果
        self.results = self.load_existing_results()
        
        logger.info(f"BVPartiBatchProcessor initialized:")
        logger.info(f"- Output directory: {self.output_dir}")
        logger.info(f"- Max workers: {self.max_workers}")
        logger.info(f"- Time limit: {self.time_limit} seconds")
        logger.info(f"- Existing results: {len(self.results)} files")
    
    def load_existing_results(self):
        """加载已有的结果文件"""
        results = {}
        if self.result_file.exists():
            try:
                with open(self.result_file, 'r') as f:
                    results = json.load(f)
                logger.info(f"Loaded {len(results)} existing results")
            except Exception as e:
                logger.warning(f"Failed to load existing results: {e}")
        return results
    
    def save_results(self):
        """保存结果到文件"""
        try:
            with open(self.result_file, 'w') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {self.result_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def process_single_file(self, file_path, timeout=None):
        """
        处理单个SMT2文件
        
        Args:
            file_path: SMT2文件路径
            timeout: 超时时间
            
        Returns:
            dict: 处理结果
        """
        if timeout is None:
            timeout = self.time_limit
        
        file_path = str(file_path)
        filename = os.path.basename(file_path)
        
        # 检查是否已经处理过
        if filename in self.results:
            logger.debug(f"Skipping already processed file: {filename}")
            return self.results[filename]
        
        logger.info(f"Processing: {filename}")
        
        try:
            # 初始化求解器
            config = BVPartiConfig()
            config.time_limit = timeout
            solver = BVPartiSolver(config)
            
            # 记录开始时间
            start_time = time.time()
            
            # 求解
            result, exec_time, stdout, stderr = solver.solve_smt2_file(file_path, timeout)
            
            # 记录结束时间
            end_time = time.time()
            total_time = end_time - start_time
            
            # 构建结果
            result_data = {
                'filename': filename,
                'file_path': file_path,
                'result': result,
                'execution_time': exec_time,
                'total_time': total_time,
                'timeout': timeout,
                'timestamp': datetime.now().isoformat(),
                'stdout': stdout,
                'stderr': stderr,
                'success': result in ['sat', 'unsat']
            }
            
            # 清理求解器
            solver.cleanup_all_temp_folders()
            
            logger.info(f"Completed: {filename} -> {result} ({exec_time:.2f}s)")
            return result_data
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            return {
                'filename': filename,
                'file_path': file_path,
                'result': 'error',
                'execution_time': 0,
                'total_time': 0,
                'timeout': timeout,
                'timestamp': datetime.now().isoformat(),
                'stdout': '',
                'stderr': str(e),
                'success': False
            }
    
    def process_files(self, file_paths, sample_size=None):
        """
        批量处理文件
        
        Args:
            file_paths: 文件路径列表
            sample_size: 随机采样大小，None表示处理所有文件
        """
        # 转换为Path对象
        file_paths = [Path(p) for p in file_paths if Path(p).exists()]
        
        if not file_paths:
            logger.warning("No valid files to process")
            return
        
        # 随机采样
        if sample_size and sample_size < len(file_paths):
            import random
            file_paths = random.sample(file_paths, sample_size)
            logger.info(f"Randomly selected {sample_size} files from {len(file_paths)} total files")
        
        logger.info(f"Processing {len(file_paths)} files with {self.max_workers} workers")
        
        # 并行处理
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交任务
            future_to_file = {
                executor.submit(self.process_single_file, file_path): file_path 
                for file_path in file_paths
            }
            
            # 收集结果
            completed = 0
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                filename = os.path.basename(str(file_path))
                
                try:
                    result_data = future.result()
                    self.results[filename] = result_data
                    completed += 1
                    
                    # 定期保存结果
                    if completed % 10 == 0:
                        self.save_results()
                        logger.info(f"Progress: {completed}/{len(file_paths)} files completed")
                        
                except Exception as e:
                    logger.error(f"Failed to get result for {filename}: {e}")
        
        # 最终保存结果
        self.save_results()
        logger.info(f"Batch processing completed: {len(self.results)} total results")
    
    def generate_summary_report(self):
        """生成汇总报告"""
        if not self.results:
            logger.warning("No results to summarize")
            return
        
        # 统计结果
        stats = {
            'total_files': len(self.results),
            'sat': 0,
            'unsat': 0,
            'timeout': 0,
            'unknown': 0,
            'error': 0,
            'success_rate': 0,
            'avg_time': 0,
            'total_time': 0
        }
        
        times = []
        for result in self.results.values():
            result_type = result.get('result', 'unknown')
            if result_type in stats:
                stats[result_type] += 1
            
            exec_time = result.get('execution_time', 0)
            if exec_time > 0:
                times.append(exec_time)
            stats['total_time'] += exec_time
        
        # 计算统计信息
        stats['success_rate'] = (stats['sat'] + stats['unsat']) / stats['total_files'] * 100
        stats['avg_time'] = sum(times) / len(times) if times else 0
        
        # 保存报告
        report_file = self.output_dir / "summary_report.json"
        with open(report_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # 打印报告
        logger.info("=== BVParti Batch Processing Summary ===")
        logger.info(f"Total files: {stats['total_files']}")
        logger.info(f"SAT: {stats['sat']}")
        logger.info(f"UNSAT: {stats['unsat']}")
        logger.info(f"Timeout: {stats['timeout']}")
        logger.info(f"Unknown: {stats['unknown']}")
        logger.info(f"Error: {stats['error']}")
        logger.info(f"Success rate: {stats['success_rate']:.1f}%")
        logger.info(f"Average time: {stats['avg_time']:.2f}s")
        logger.info(f"Total time: {stats['total_time']:.2f}s")
        logger.info(f"Report saved to: {report_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="BVParti Batch Processor")
    parser.add_argument('input_path', help="Input directory or file pattern")
    parser.add_argument('--output-dir', default='batch_output', help="Output directory")
    parser.add_argument('--max-workers', type=int, default=4, help="Maximum number of worker processes")
    parser.add_argument('--time-limit', type=int, default=300, help="Time limit per instance (seconds)")
    parser.add_argument('--sample-size', type=int, help="Random sample size")
    parser.add_argument('--pattern', default='*.smt2', help="File pattern to match")
    
    args = parser.parse_args()
    
    # 设置日志
    logger.add(f"{args.output_dir}/logs/batch_bvparti_{{time}}.log", 
               rotation="500 MB", retention="10 days")
    
    # 获取文件列表
    input_path = Path(args.input_path)
    if input_path.is_file():
        file_paths = [input_path]
    elif input_path.is_dir():
        file_paths = list(input_path.glob(args.pattern))
    else:
        # 尝试作为glob模式
        file_paths = glob.glob(str(input_path))
    
    if not file_paths:
        logger.error(f"No files found matching: {args.input_path}")
        return 1
    
    logger.info(f"Found {len(file_paths)} files to process")
    
    # 初始化批处理器
    processor = BVPartiBatchProcessor(
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        time_limit=args.time_limit
    )
    
    try:
        # 处理文件
        processor.process_files(file_paths, sample_size=args.sample_size)
        
        # 生成报告
        processor.generate_summary_report()
        
        logger.info("Batch processing completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
