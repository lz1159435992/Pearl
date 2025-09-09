import os
import sys

# Add the project root to the Python path to resolve module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import datetime
from multiprocessing import Process, Manager, Queue, set_start_method
import json
import random
import re
import time
import traceback
import argparse
from loguru import logger
import torch

# Import necessary components from the existing framework
from test_rl.test_cvc5.predict_z3_process.run_predictor import (
    setup_logger,
    load_dictionary,
    get_solver,
    process_single_file_with_timeout,
    EnhancedEightClassModel,
    SimpleClassifier,
    Z3Solver,
    CVC5Solver,
    MathSAT5Solver
)
from test_rl.predictor.bert_embedder_test import CodeEmbedder_normalize
from test_rl.test_script.utils import normalize_smt_str


def analyze_time_distribution(file_path):
    """
    Analyzes and prints the solving time distribution for non-unsat constraints.
    """
    logger.info(f"Starting time distribution analysis for: {file_path}")
    
    data = load_dictionary(file_path)
    if not data:
        logger.warning("Data file is empty or could not be loaded. Aborting analysis.")
        return

    time_bins = {}
    total_non_unsat = 0

    for key, value in data.items():
        if isinstance(value, list) and len(value) > 1:
            result, time_taken = value[0], value[1]
            if result != 'unsat':
                total_non_unsat += 1
                if isinstance(time_taken, (int, float)):
                    # Determine the time bin (e.g., 0-100, 100-200, etc.)
                    bin_index = int(time_taken // 100)
                    bin_label = f"{bin_index * 100}s - {(bin_index + 1) * 100}s"
                    
                    if bin_label not in time_bins:
                        time_bins[bin_label] = 0
                    time_bins[bin_label] += 1
    
    logger.info("=" * 80)
    logger.info("Solving Time Distribution for Non-UNSAT Constraints")
    logger.info("=" * 80)
    
    if not time_bins:
        logger.info("No non-UNSAT constraints with valid times found.")
        return
        
    # Sort bins by the starting time for clean output
    sorted_bins = sorted(time_bins.items(), key=lambda item: int(item[0].split('s')[0]))

    for bin_label, count in sorted_bins:
        logger.info(f"{bin_label}: {count} constraints")
        
    logger.info("-" * 80)
    logger.info(f"Total non-UNSAT constraints analyzed: {total_non_unsat}")
    logger.info("=" * 80)


def run_advanced_prediction_flow(args):
    """
    Main function to run the advanced, multi-stage prediction and solving workflow.
    """
    logger, batch_id = setup_logger()
    args.batch_id = batch_id
    logger.info("=" * 80)
    logger.info(f"Advanced SMT Solver (Multi-Stage Prediction) - Start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Random Samples: {args.num_samples}, Prediction Threshold: {args.prediction_threshold}")
    logger.info("=" * 80)

    # 1. Load all data sources and models
    try:
        logger.info("Loading data sources...")
        # Use the robust load_dictionary function for all data files
        source_constraints = load_dictionary(args.source_constraints_path)
        direct_solve_cache = load_dictionary(args.direct_solve_cache_path)
        rl_solve_cache = load_dictionary(args.rl_solve_cache_path)

        if os.path.exists(args.output_path):
            output_dict = load_dictionary(args.output_path)
            logger.info(f"Loaded existing output file with {len(output_dict)} entries.")
        else:
            output_dict = {}
            logger.info("No existing output file found. Starting fresh.")

        logger.info("Loading predictor models...")
        solvability_predictor = SimpleClassifier()
        solvability_predictor.load_state_dict(torch.load(args.binary_model_path))
        solvability_predictor.eval()

        time_predictor = EnhancedEightClassModel()
        time_predictor.load_state_dict(torch.load(args.eight_class_model_path))
        time_predictor.eval()
        
        embedder = CodeEmbedder_normalize()
        logger.info("All data and models loaded successfully.")
    except Exception as e:
        logger.error(f"Initialization failed: Could not load data or models. Error: {e}")
        return

    # 2. Sampling Logic (Conditional)
    unprocessed_keys = [key for key in source_constraints.keys() if key not in output_dict]
    if not unprocessed_keys:
        logger.info("All constraints have already been processed. Exiting.")
        return
    
    # If --num_samples is 0, process all available files. Otherwise, sample a subset.
    if args.num_samples == 0:
        selected_keys = unprocessed_keys
        logger.info(f"Processing all {len(selected_keys)} new constraints as --num_samples is 0.")
    else:
        num_to_sample = min(args.num_samples, len(unprocessed_keys))
        selected_keys = random.sample(unprocessed_keys, num_to_sample)
        logger.info(f"Randomly selected {num_to_sample} new constraints to process.")

    # 3. Loop and execute conditional logic
    for i, key in enumerate(selected_keys):
        logger.info("-" * 80)
        logger.info(f"Processing file {i+1}/{len(selected_keys)}: {key}")

        try:
            # Prediction Phase
            with open(key, 'r') as f:
                smtlib_str_raw = f.read()
            dict_obj = json.loads(smtlib_str_raw)
            smtlib_str = dict_obj.get('smt_script') or dict_obj.get('script')
            
            normalized_str, var_dict, _ = normalize_smt_str(smtlib_str)
            embedding = embedder.get_max_pooling_embedding(normalized_str, var_dict)

            with torch.no_grad():
                solvability_output = solvability_predictor(embedding)
                is_solvable = (solvability_output > 0.5).int().item() == 1

                time_output = time_predictor(embedding)
                _, predicted_time_class = torch.max(time_output, 1)
                time_class = predicted_time_class.item()

            logger.info(f"Prediction - Solvable: {is_solvable}, Time Class: {time_class}")

            # Conditional Execution
            # Path 1: Easy/Unsolvable -> Use direct solve cache
            if not is_solvable or time_class <= args.prediction_threshold:
                logger.info(f"Decision: Path 1 (Easy/Unsolvable). Reading from direct solve cache.")
                if key in direct_solve_cache:
                    # Expand the cached result to the full 10-element format for consistency
                    cached_result = direct_solve_cache[key]
                    
                    # Ensure the cached_result is a list before accessing indices
                    if not isinstance(cached_result, list):
                        cached_result = [str(cached_result)]

                    result_list = [
                        cached_result[0] if len(cached_result) > 0 else 'unknown',  # Original solver result
                        cached_result[1] if len(cached_result) > 1 else 0,          # Original solver time
                        cached_result[2] if len(cached_result) > 2 else 'N/A',      # Original memory usage
                        0,                                                         # Total execution time for this script (placeholder)
                        cached_result[1] if len(cached_result) > 1 else 0,          # Cumulative solver time (same as original)
                        0,                                                         # Final successful solve time (not applicable)
                        0,                                                         # Cumulative LLM time (not applicable)
                        'cached_direct_solve',                                     # Status
                        [],                                                        # Final successful assignment (not applicable)
                        []                                                         # History of all assignments (not applicable)
                    ]
                    output_dict[key] = result_list
                    logger.info("Found and recorded result from direct solve cache with standardized format.")
                else:
                    logger.warning("Constraint not found in direct solve cache. Skipping.")
                continue

            # Path 2: Hard but already in RL cache
            logger.info("Decision: Path 2/3 (Hard Problem). Checking RL solve cache...")
            if key in rl_solve_cache:
                output_dict[key] = rl_solve_cache[key]
                logger.info("Found and recorded result from RL solve cache.")
                continue
            
            # Path 3: New Hard Problem -> Run RL+LLM Solver
            logger.info("Decision: Path 3 (New Hard Problem). Not in RL cache. Invoking RL+LLM solver.")
            # We need a placeholder for the `list1` argument for the solver function
            placeholder_list1 = ["", 0, ""] 
            process_single_file_with_timeout(key, placeholder_list1, output_dict, args)
            logger.info("RL+LLM solver finished for this file.")

        except Exception as e:
            logger.error(f"An error occurred while processing {key}: {e}")
            traceback.print_exc()
        finally:
            # Save progress after each file
            with open(args.output_path, 'w') as f:
                json.dump(output_dict, f, indent=4)
    
    logger.info("=" * 80)
    logger.info(f"Advanced Prediction Workflow Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total entries in output file: {len(output_dict)}")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Advanced SMT Solver with Multi-Stage Prediction")
    
    # --- Paths to Data ---
    parser.add_argument('--source_constraints_path', type=str, 
                        default='/home/lz/sibyl_3/src/networks/info_dict_rl.txt',
                        help='Path to the JSON file with all constraint file paths as keys.')
    parser.add_argument('--direct_solve_cache_path', type=str,
                        default='/home/lz/PycharmProjects/Pearl/test_rl/test_solve/info_dict_bingxing.txt',
                        help='Path to the cache of pre-solved "easy" problems.')
    parser.add_argument('--rl_solve_cache_path', type=str,
                        default='/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1110_pre_SMTimer_llama3.1:70b_1200s_info_dict_rl.txt',
                        help='Path to the cache of pre-solved "hard" problems using RL.')
    parser.add_argument('--output_path', type=str,
                        default='advanced_solver_results_all.json',
                        help='Path to save the final results.')

    # --- Analysis Flag ---
    parser.add_argument('--analyze', action='store_true',
                        help='If set, run the time distribution analysis on the direct_solve_cache and exit.')

    # --- Control Flow Arguments ---
    parser.add_argument('--num_samples', type=int, default=0,
                        help='Number of new constraints to randomly select and process. Set to 0 to process all.')
    parser.add_argument('--prediction_threshold', type=int, default=5,
                        # 500s
                        help='Predicted time class threshold (inclusive). Problems <= this value are considered "easy".')

    # --- RL+LLM Solver Arguments (inherited from run_predictor.py) ---
    parser.add_argument('--binary_model_path', type=str, default='/home/lz/PycharmProjects/Pearl/test_rl/test_cvc5/predict_z3_process/models/bert_predictor_mask_best.pth')
    parser.add_argument('--eight_class_model_path', type=str, default='/home/lz/PycharmProjects/Pearl/test_rl/test_cvc5/predict_z3_process/models/bert_predictor_2_mask_best_model.pth')
    parser.add_argument('--timeout', type=int, default=1200, help='Timeout for the RL+LLM solver in seconds.')
    parser.add_argument('--num_episodes', type=int, default=1)
    parser.add_argument('--record_period', type=int, default=1)
    parser.add_argument('--llm_host', type=str, default='http://172.29.7.221:32783')
    parser.add_argument('--llm_model', type=str, default='llama3.1:70b')
    parser.add_argument('--solver', type=str, default='z3', choices=['z3', 'cvc5', 'mathsat'])

    #QF_NIA

    parser.add_argument('--source_constraints_path', type=str, 
                        default='/home/lz/PycharmProjects/Pearl/test_rl/predictor/smt_comp_NIA/QF_NIA_test.json',
                        help='Path to the JSON file with all constraint file paths as keys.')
    parser.add_argument('--direct_solve_cache_path', type=str,
                        default='/home/lz/PycharmProjects/Pearl/test_rl/test_solve/NIA/NIA.json',
                        help='Path to the cache of pre-solved "easy" problems.')
    parser.add_argument('--rl_solve_cache_path', type=str,
                        default='/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_0503_pre_llm_llama3.1:70b_1200s_QF_NIA.txt',
                        help='Path to the cache of pre-solved "hard" problems using RL.')
    parser.add_argument('--output_path', type=str,
                        default='QF_NIA_advanced_solver_results_all.json',
                        help='Path to save the final results.')

    # --- Analysis Flag ---
    parser.add_argument('--analyze', action='store_true',
                        help='If set, run the time distribution analysis on the direct_solve_cache and exit.')

    # --- Control Flow Arguments ---
    parser.add_argument('--num_samples', type=int, default=0,
                        help='Number of new constraints to randomly select and process. Set to 0 to process all.')
    parser.add_argument('--prediction_threshold', type=int, default=5,
                        # 500s
                        help='Predicted time class threshold (inclusive). Problems <= this value are considered "easy".')

    # --- RL+LLM Solver Arguments (inherited from run_predictor.py) ---
    parser.add_argument('--binary_model_path', type=str, default='/home/lz/PycharmProjects/Pearl/test_rl/test_cvc5/predict_z3_process/models/bert_predictor_mask_best.pth')
    parser.add_argument('--eight_class_model_path', type=str, default='/home/lz/PycharmProjects/Pearl/test_rl/test_cvc5/predict_z3_process/models/bert_predictor_2_mask_best_model.pth')
    parser.add_argument('--timeout', type=int, default=1200, help='Timeout for the RL+LLM solver in seconds.')
    parser.add_argument('--num_episodes', type=int, default=1)
    parser.add_argument('--record_period', type=int, default=1)
    parser.add_argument('--llm_host', type=str, default='http://172.29.7.221:32943')
    parser.add_argument('--llm_model', type=str, default='llama3.1:70b')
    parser.add_argument('--solver', type=str, default='z3', choices=['z3', 'cvc5', 'mathsat'])
    args = parser.parse_args()

    
    # If the analyze flag is set, run the analysis and exit.
    if args.analyze:
        analyze_time_distribution(args.direct_solve_cache_path)
    else:
        run_advanced_prediction_flow(args)

if __name__ == "__main__":
    main() 