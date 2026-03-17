import os
import sys
import re
import json
import time
import argparse
import datetime
from typing import Dict, List, Tuple, Optional
from loguru import logger
import torch
import numpy as np

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import necessary components
from test_rl.test_cvc5.predict_z3_process.run_advanced_predictor import (
    setup_logger, load_dictionary, get_solver, process_single_file_with_timeout,
    EnhancedEightClassModel, SimpleClassifier, Z3Solver, CVC5Solver, MathSAT5Solver
)
from test_rl.predictor.bert_embedder_test import CodeEmbedder_normalize
from test_rl.test_script.utils import normalize_smt_str


class HybridScreeningStrategy:
    """
    混合策略筛选器 - 结合多种方法识别困难约束
    """
    
    def __init__(self, args):
        self.args = args
        self.logger, self.batch_id = setup_logger()
        
        # 加载模型和工具
        self._load_models()
        self._load_solvers()
        
        # 筛选策略权重
        self.weights = {
            'feature_score': 0.4,
            'quick_solve': 0.3,
            'prediction_score': 0.3
        }
        
        # 统计信息
        self.stats = {
            'total_constraints': 0,
            'feature_screened': 0,
            'quick_solve_screened': 0,
            'prediction_screened': 0,
            'hybrid_screened': 0,
            'direct_solved': 0,
            'rl_llm_solved': 0
        }
    
    def _load_models(self):
        """加载预测模型"""
        try:
            self.logger.info("Loading prediction models...")
            
            # 可解性预测模型
            self.solvability_predictor = SimpleClassifier()
            self.solvability_predictor.load_state_dict(
                torch.load(self.args.binary_model_path)
            )
            self.solvability_predictor.eval()
            
            # 时间预测模型
            self.time_predictor = EnhancedEightClassModel()
            self.time_predictor.load_state_dict(
                torch.load(self.args.eight_class_model_path)
            )
            self.time_predictor.eval()
            
            # 嵌入器
            self.embedder = CodeEmbedder_normalize()
            
            self.logger.info("All models loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            raise
    
    def _load_solvers(self):
        """加载求解器"""
        try:
            self.logger.info("Loading solvers...")
            self.quick_solver = get_solver(self.args.solver)
            self.logger.info(f"Quick solver ({self.args.solver}) loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load solver: {e}")
            raise
    
    def extract_constraint_features(self, smtlib_str: str) -> Dict[str, float]:
        """
        提取约束的结构特征
        """
        features = {
            'variable_count': 0,
            'constraint_count': 0,
            'nonlinear_ops': 0,
            'modular_ops': 0,
            'bitvector_width': 0,
            'nesting_depth': 0,
            'complexity_score': 0
        }
        
        try:
            # 变量数量
            var_pattern = r'declare-\w+\s+(\w+)'
            variables = set(re.findall(var_pattern, smtlib_str))
            features['variable_count'] = len(variables)
            
            # 约束数量
            assert_pattern = r'assert'
            features['constraint_count'] = len(re.findall(assert_pattern, smtlib_str))
            
            # 非线性操作
            nonlinear_patterns = [
                r'\*\*',  # 幂运算
                r'\*',    # 乘法
                r'/',     # 除法
                r'%',     # 模运算
                r'sqrt',  # 平方根
                r'pow'    # 幂函数
            ]
            for pattern in nonlinear_patterns:
                features['nonlinear_ops'] += len(re.findall(pattern, smtlib_str))
            
            # 模运算
            mod_pattern = r'%'
            features['modular_ops'] = len(re.findall(mod_pattern, smtlib_str))
            
            # 位向量宽度
            bv_pattern = r'_ bv\d+ (\d+)'
            bv_widths = re.findall(bv_pattern, smtlib_str)
            if bv_widths:
                features['bitvector_width'] = max(int(w) for w in bv_widths)
            
            # 嵌套深度（简化计算）
            let_pattern = r'let'
            features['nesting_depth'] = len(re.findall(let_pattern, smtlib_str))
            
            # 复杂度评分
            features['complexity_score'] = (
                features['variable_count'] * 0.3 +
                features['nonlinear_ops'] * 0.4 +
                features['modular_ops'] * 0.3 +
                features['nesting_depth'] * 0.2 +
                features['bitvector_width'] * 0.1
            )
            
        except Exception as e:
            self.logger.warning(f"Error extracting features: {e}")
        
        return features
    
    def feature_based_screening(self, smtlib_str: str) -> Tuple[bool, float]:
        """
        基于特征的筛选
        """
        features = self.extract_constraint_features(smtlib_str)
        
        # 特征评分（0-1之间）
        feature_score = min(features['complexity_score'] / 100.0, 1.0)
        
        # 阈值判断
        is_difficult = feature_score > 0.6
        
        self.stats['feature_screened'] += 1 if is_difficult else 0
        
        return is_difficult, feature_score
    
    def quick_solve_screening(self, smtlib_str: str) -> Tuple[bool, float]:
        """
        快速求解测试筛选
        """
        try:
            # 短时间求解测试
            start_time = time.time()
            result = self.quick_solver.solve(smtlib_str, timeout=5)  # 5秒超时
            solve_time = time.time() - start_time
            
            # 判断是否为困难约束
            is_difficult = (
                result.result == 'unknown' or 
                (result.result == 'sat' and solve_time > 3.0) or
                (result.result == 'unsat' and solve_time > 3.0)
            )
            
            # 评分（基于求解时间和结果）
            if result.result == 'unknown':
                quick_score = 1.0
            elif solve_time > 3.0:
                quick_score = 0.8
            else:
                quick_score = 0.2
            
            self.stats['quick_solve_screened'] += 1 if is_difficult else 0
            
            return is_difficult, quick_score
            
        except Exception as e:
            self.logger.warning(f"Quick solve screening failed: {e}")
            return False, 0.0
    
    def prediction_based_screening(self, smtlib_str: str) -> Tuple[bool, float]:
        """
        基于预测模型的筛选
        """
        try:
            # 标准化约束
            normalized_str, var_dict, _ = normalize_smt_str(smtlib_str)
            embedding = self.embedder.get_max_pooling_embedding(normalized_str, var_dict)
            
            with torch.no_grad():
                # 可解性预测
                solvability_output = self.solvability_predictor(embedding)
                solvability_score = torch.sigmoid(solvability_output).item()
                
                # 时间预测
                time_output = self.time_predictor(embedding)
                _, predicted_time_class = torch.max(time_output, 1)
                time_class = predicted_time_class.item()
                
                # 时间类别转换为评分
                time_score = (time_class + 1) / 8.0  # 转换为0-1范围
            
            # 综合预测评分
            prediction_score = (1 - solvability_score) * 0.6 + time_score * 0.4
            is_difficult = prediction_score > 0.6
            
            self.stats['prediction_screened'] += 1 if is_difficult else 0
            
            return is_difficult, prediction_score
            
        except Exception as e:
            self.logger.warning(f"Prediction screening failed: {e}")
            return False, 0.0
    
    def hybrid_screening(self, smtlib_str: str) -> Tuple[bool, Dict[str, float]]:
        """
        混合策略筛选
        """
        # 执行三种筛选方法
        feature_difficult, feature_score = self.feature_based_screening(smtlib_str)
        quick_difficult, quick_score = self.quick_solve_screening(smtlib_str)
        pred_difficult, pred_score = self.prediction_based_screening(smtlib_str)
        
        # 计算加权综合评分
        hybrid_score = (
            feature_score * self.weights['feature_score'] +
            quick_score * self.weights['quick_solve'] +
            pred_score * self.weights['prediction_score']
        )
        
        # 综合决策
        is_difficult = hybrid_score > 0.6
        
        self.stats['hybrid_screened'] += 1 if is_difficult else 0
        
        scores = {
            'feature_score': feature_score,
            'quick_score': quick_score,
            'prediction_score': pred_score,
            'hybrid_score': hybrid_score
        }
        
        return is_difficult, scores
    
    def solve_constraint(self, constraint_path: str, smtlib_str: str, 
                        is_difficult: bool) -> Dict:
        """
        求解约束
        """
        try:
            if is_difficult:
                # 使用RL+LLM方法
                self.logger.info(f"Using RL+LLM for difficult constraint: {constraint_path}")
                
                # 调用RL+LLM求解器
                output_dict = {}
                placeholder_list = ["", 0, ""]
                process_single_file_with_timeout(
                    constraint_path, placeholder_list, output_dict, self.args
                )
                
                if constraint_path in output_dict:
                    result = output_dict[constraint_path]
                    self.stats['rl_llm_solved'] += 1
                    return {
                        'method': 'rl_llm',
                        'result': result[0] if len(result) > 0 else 'unknown',
                        'time': result[1] if len(result) > 1 else 0,
                        'status': result[7] if len(result) > 7 else 'unknown'
                    }
                else:
                    return {'method': 'rl_llm', 'result': 'error', 'time': 0, 'status': 'error'}
            else:
                # 使用直接求解
                self.logger.info(f"Using direct solve for easy constraint: {constraint_path}")
                
                start_time = time.time()
                result = self.quick_solver.solve(smtlib_str, timeout=self.args.timeout)
                solve_time = time.time() - start_time
                
                self.stats['direct_solved'] += 1
                return {
                    'method': 'direct',
                    'result': result.result,
                    'time': solve_time,
                    'status': 'direct_solve'
                }
                
        except Exception as e:
            self.logger.error(f"Error solving constraint {constraint_path}: {e}")
            return {'method': 'error', 'result': 'error', 'time': 0, 'status': 'error'}
    
    def run_rq5_experiment(self, constraint_paths: List[str]) -> Dict:
        """
        运行RQ5实验
        """
        self.logger.info("=" * 80)
        self.logger.info("RQ5 Hybrid Screening Experiment")
        self.logger.info("=" * 80)
        
        results = {}
        self.stats['total_constraints'] = len(constraint_paths)
        
        for i, constraint_path in enumerate(constraint_paths):
            self.logger.info(f"Processing {i+1}/{len(constraint_paths)}: {constraint_path}")
            
            try:
                # 读取约束
                with open(constraint_path, 'r') as f:
                    smtlib_str_raw = f.read()
                
                dict_obj = json.loads(smtlib_str_raw)
                smtlib_str = dict_obj.get('smt_script') or dict_obj.get('script')
                
                # 混合筛选
                is_difficult, scores = self.hybrid_screening(smtlib_str)
                
                # 求解约束
                solve_result = self.solve_constraint(constraint_path, smtlib_str, is_difficult)
                
                # 记录结果
                results[constraint_path] = {
                    'is_difficult': is_difficult,
                    'scores': scores,
                    'solve_result': solve_result
                }
                
                self.logger.info(f"Decision: {'Difficult' if is_difficult else 'Easy'}, "
                               f"Method: {solve_result['method']}, "
                               f"Result: {solve_result['result']}, "
                               f"Time: {solve_result['time']:.2f}s")
                
            except Exception as e:
                self.logger.error(f"Error processing {constraint_path}: {e}")
                results[constraint_path] = {
                    'is_difficult': False,
                    'scores': {},
                    'solve_result': {'method': 'error', 'result': 'error', 'time': 0, 'status': 'error'}
                }
        
        return results
    
    def analyze_results(self, results: Dict) -> Dict:
        """
        分析实验结果
        """
        analysis = {
            'screening_stats': self.stats.copy(),
            'method_performance': {},
            'time_distribution': {},
            'accuracy_metrics': {}
        }
        
        # 方法性能分析
        method_stats = {'direct': [], 'rl_llm': []}
        for constraint_path, result in results.items():
            method = result['solve_result']['method']
            if method in method_stats:
                method_stats[method].append({
                    'time': result['solve_result']['time'],
                    'result': result['solve_result']['result']
                })
        
        for method, data in method_stats.items():
            if data:
                times = [d['time'] for d in data if isinstance(d['time'], (int, float))]
                sat_count = sum(1 for d in data if d['result'] == 'sat')
                
                analysis['method_performance'][method] = {
                    'count': len(data),
                    'avg_time': np.mean(times) if times else 0,
                    'success_rate': sat_count / len(data) if data else 0
                }
        
        # 时间分布分析
        all_times = []
        for result in results.values():
            time_val = result['solve_result']['time']
            if isinstance(time_val, (int, float)) and time_val > 0:
                all_times.append(time_val)
        
        if all_times:
            analysis['time_distribution'] = {
                'mean': np.mean(all_times),
                'median': np.median(all_times),
                'std': np.std(all_times),
                'min': np.min(all_times),
                'max': np.max(all_times)
            }
        
        return analysis
    
    def print_summary(self, analysis: Dict):
        """
        打印实验总结
        """
        self.logger.info("=" * 80)
        self.logger.info("RQ5 Experiment Summary")
        self.logger.info("=" * 80)
        
        stats = analysis['screening_stats']
        self.logger.info(f"Total constraints: {stats['total_constraints']}")
        self.logger.info(f"Feature screened: {stats['feature_screened']} ({stats['feature_screened']/stats['total_constraints']*100:.1f}%)")
        self.logger.info(f"Quick solve screened: {stats['quick_solve_screened']} ({stats['quick_solve_screened']/stats['total_constraints']*100:.1f}%)")
        self.logger.info(f"Prediction screened: {stats['prediction_screened']} ({stats['prediction_screened']/stats['total_constraints']*100:.1f}%)")
        self.logger.info(f"Hybrid screened: {stats['hybrid_screened']} ({stats['hybrid_screened']/stats['total_constraints']*100:.1f}%)")
        self.logger.info(f"Direct solved: {stats['direct_solved']}")
        self.logger.info(f"RL+LLM solved: {stats['rl_llm_solved']}")
        
        # 方法性能
        for method, perf in analysis['method_performance'].items():
            self.logger.info(f"{method.upper()} - Count: {perf['count']}, "
                           f"Avg Time: {perf['avg_time']:.2f}s, "
                           f"Success Rate: {perf['success_rate']:.2f}")
        
        # 时间分布
        if analysis['time_distribution']:
            time_dist = analysis['time_distribution']
            self.logger.info(f"Time Distribution - Mean: {time_dist['mean']:.2f}s, "
                           f"Median: {time_dist['median']:.2f}s, "
                           f"Std: {time_dist['std']:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="RQ5 Hybrid Screening Experiment")
    
    # 数据路径
    parser.add_argument('--source_constraints_path', type=str, 
                        default='/home/lz/sibyl_3/src/networks/info_dict_rl.txt',
                        help='Path to source constraints')
    parser.add_argument('--output_path', type=str,
                        default='rq5_hybrid_results.json',
                        help='Path to save results')
    
    # 模型路径
    parser.add_argument('--binary_model_path', type=str, 
                        default='/home/lz/PycharmProjects/Pearl/test_rl/test_cvc5/predict_z3_process/models/bert_predictor_mask_best.pth')
    parser.add_argument('--eight_class_model_path', type=str, 
                        default='/home/lz/PycharmProjects/Pearl/test_rl/test_cvc5/predict_z3_process/models/bert_predictor_2_mask_best_model.pth')
    
    # 求解器配置
    parser.add_argument('--solver', type=str, default='z3', choices=['z3', 'cvc5', 'mathsat'])
    parser.add_argument('--timeout', type=int, default=1200)
    
    # 实验控制
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of constraints to test (0 for all)')
    parser.add_argument('--random_seed', type=int, default=42)
    
    # RL+LLM配置
    parser.add_argument('--llm_host', type=str, default='http://172.29.7.221:32943')
    parser.add_argument('--llm_model', type=str, default='llama3.1:70b')
    parser.add_argument('--num_episodes', type=int, default=1)
    parser.add_argument('--record_period', type=int, default=1)
    
    args = parser.parse_args()
    
    # 设置随机种子
    import random
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # 创建混合策略实验器
    experiment = HybridScreeningStrategy(args)
    
    # 加载约束数据
    source_constraints = load_dictionary(args.source_constraints_path)
    constraint_paths = list(source_constraints.keys())
    
    # 采样约束
    if args.num_samples > 0 and args.num_samples < len(constraint_paths):
        constraint_paths = random.sample(constraint_paths, args.num_samples)
    
    # 运行实验
    results = experiment.run_rq5_experiment(constraint_paths)
    
    # 分析结果
    analysis = experiment.analyze_results(results)
    
    # 打印总结
    experiment.print_summary(analysis)
    
    # 保存结果
    with open(args.output_path, 'w') as f:
        json.dump({
            'results': results,
            'analysis': analysis,
            'config': vars(args)
        }, f, indent=4)
    
    experiment.logger.info(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main() 