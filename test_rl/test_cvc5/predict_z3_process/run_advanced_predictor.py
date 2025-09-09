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

    # 2. Random Sampling
    unprocessed_keys = [key for key in source_constraints.keys() if key not in output_dict]
    if not unprocessed_keys:
        logger.info("All constraints have already been processed. Exiting.")
        return
    
    num_to_sample = min(args.num_samples, len(unprocessed_keys))
    selected_keys = random.sample(unprocessed_keys, num_to_sample)
    logger.info(f"Randomly selected {num_to_sample} new constraints to process.")

    # 3. Loop and execute conditional logic
    for i, key in enumerate(selected_keys):
        logger.info("-" * 80)
        logger.info(f"Processing file {i+1}/{num_to_sample}: {key}")

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
                    output_dict[key] = direct_solve_cache[key]
                    logger.info("Found and recorded result from direct solve cache.")
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
                        default='/home/lz/constraint_solve_file/smtimer-533-result.txt',
                        help='Path to the cache of pre-solved "easy" problems.')
    parser.add_argument('--rl_solve_cache_path', type=str,
                        default='/home/lz/PycharmProjects/Pearl/test_rl/info_dict_gai_6_normal_1110_pre_SMTimer_llama3.1:70b_1200s_info_dict_rl.txt',
                        help='Path to the cache of pre-solved "hard" problems using RL.')
    parser.add_argument('--output_path', type=str,
                        default='advanced_solver_results.json',
                        help='Path to save the final results.')

    # --- Control Flow Arguments ---
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Number of new constraints to randomly select and process.')
    parser.add_argument('--prediction_threshold', type=int, default=5,
                        # 500s
                        help='Predicted time class threshold (inclusive). Problems <= this value are considered "easy".')

    # --- RL+LLM Solver Arguments (inherited from run_predictor.py) ---
    parser.add_argument('--binary_model_path', type=str, default='models/binary_classifier.pth')
    parser.add_argument('--eight_class_model_path', type=str, default='models/eight_class_model.pth')
    parser.add_argument('--timeout', type=int, default=1200, help='Timeout for the RL+LLM solver in seconds.')
    parser.add_argument('--num_episodes', type=int, default=1)
    parser.add_argument('--record_period', type=int, default=1)
    parser.add_argument('--llm_host', type=str, default='http://172.29.7.221:32943')
    parser.add_argument('--llm_model', type=str, default='llama3.1:70b')
    parser.add_argument('--solver', type=str, default='z3', choices=['z3', 'cvc5', 'mathsat'])

    args = parser.parse_args()
    
    run_advanced_prediction_flow(args)

if __name__ == "__main__":
    main() 