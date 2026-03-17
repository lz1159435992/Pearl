#!/usr/bin/env python3
"""
RQ5 Hybrid Screening Strategy - Usage Example

This script demonstrates how to use the hybrid screening strategy for RQ5 experiments.
"""

import os
import sys
import json
import random

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from rq5_hybrid_screening import HybridScreeningStrategy, main


def run_small_experiment():
    """
    运行小规模实验示例
    """
    print("=" * 80)
    print("RQ5 Hybrid Screening - Small Scale Experiment")
    print("=" * 80)
    
    # 实验参数
    args = type('Args', (), {
        'source_constraints_path': '/home/lz/sibyl_3/src/networks/info_dict_rl.txt',
        'output_path': 'rq5_small_experiment.json',
        'binary_model_path': '/home/lz/PycharmProjects/Pearl/test_rl/test_cvc5/predict_z3_process/models/bert_predictor_mask_best.pth',
        'eight_class_model_path': '/home/lz/PycharmProjects/Pearl/test_rl/test_cvc5/predict_z3_process/models/bert_predictor_2_mask_best_model.pth',
        'solver': 'z3',
        'timeout': 1200,
        'num_samples': 20,  # 小规模测试
        'random_seed': 42,
        'llm_host': 'http://172.29.7.221:32943',
        'llm_model': 'llama3.1:70b',
        'num_episodes': 1,
        'record_period': 1
    })()
    
    # 创建实验器
    experiment = HybridScreeningStrategy(args)
    
    # 加载约束数据
    from test_rl.test_cvc5.predict_z3_process.run_advanced_predictor import load_dictionary
    source_constraints = load_dictionary(args.source_constraints_path)
    constraint_paths = list(source_constraints.keys())
    
    # 随机采样
    if args.num_samples > 0 and args.num_samples < len(constraint_paths):
        constraint_paths = random.sample(constraint_paths, args.num_samples)
    
    print(f"Selected {len(constraint_paths)} constraints for testing")
    
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
    
    print(f"\nResults saved to {args.output_path}")
    return results, analysis


def analyze_screening_accuracy(results):
    """
    分析筛选准确性
    """
    print("\n" + "=" * 80)
    print("Screening Accuracy Analysis")
    print("=" * 80)
    
    # 统计筛选结果
    total = len(results)
    difficult_count = sum(1 for r in results.values() if r['is_difficult'])
    easy_count = total - difficult_count
    
    print(f"Total constraints: {total}")
    print(f"Classified as difficult: {difficult_count} ({difficult_count/total*100:.1f}%)")
    print(f"Classified as easy: {easy_count} ({easy_count/total*100:.1f}%)")
    
    # 分析求解结果
    rl_llm_results = [r for r in results.values() if r['solve_result']['method'] == 'rl_llm']
    direct_results = [r for r in results.values() if r['solve_result']['method'] == 'direct']
    
    print(f"\nRL+LLM solved: {len(rl_llm_results)}")
    print(f"Direct solved: {len(direct_results)}")
    
    # 成功率分析
    if rl_llm_results:
        rl_llm_success = sum(1 for r in rl_llm_results if r['solve_result']['result'] == 'sat')
        rl_llm_success_rate = rl_llm_success / len(rl_llm_results)
        print(f"RL+LLM success rate: {rl_llm_success_rate:.2f}")
    
    if direct_results:
        direct_success = sum(1 for r in direct_results if r['solve_result']['result'] == 'sat')
        direct_success_rate = direct_success / len(direct_results)
        print(f"Direct solve success rate: {direct_success_rate:.2f}")


def compare_screening_methods(results):
    """
    比较不同筛选方法的效果
    """
    print("\n" + "=" * 80)
    print("Screening Method Comparison")
    print("=" * 80)
    
    # 分析每种筛选方法的评分分布
    feature_scores = []
    quick_scores = []
    prediction_scores = []
    hybrid_scores = []
    
    for result in results.values():
        scores = result['scores']
        if 'feature_score' in scores:
            feature_scores.append(scores['feature_score'])
        if 'quick_score' in scores:
            quick_scores.append(scores['quick_score'])
        if 'prediction_score' in scores:
            prediction_scores.append(scores['prediction_score'])
        if 'hybrid_score' in scores:
            hybrid_scores.append(scores['hybrid_score'])
    
    print("Score Statistics:")
    if feature_scores:
        print(f"Feature scores - Mean: {sum(feature_scores)/len(feature_scores):.3f}, "
              f"Max: {max(feature_scores):.3f}, Min: {min(feature_scores):.3f}")
    if quick_scores:
        print(f"Quick solve scores - Mean: {sum(quick_scores)/len(quick_scores):.3f}, "
              f"Max: {max(quick_scores):.3f}, Min: {min(quick_scores):.3f}")
    if prediction_scores:
        print(f"Prediction scores - Mean: {sum(prediction_scores)/len(prediction_scores):.3f}, "
              f"Max: {max(prediction_scores):.3f}, Min: {min(prediction_scores):.3f}")
    if hybrid_scores:
        print(f"Hybrid scores - Mean: {sum(hybrid_scores)/len(hybrid_scores):.3f}, "
              f"Max: {max(hybrid_scores):.3f}, Min: {min(hybrid_scores):.3f}")


def main_example():
    """
    主示例函数
    """
    print("Starting RQ5 Hybrid Screening Example...")
    
    # 运行小规模实验
    results, analysis = run_small_experiment()
    
    # 分析筛选准确性
    analyze_screening_accuracy(results)
    
    # 比较筛选方法
    compare_screening_methods(results)
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main_example() 