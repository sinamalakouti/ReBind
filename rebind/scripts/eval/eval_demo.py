#!/usr/bin/env python
import argparse
import os
from pathlib import Path
import torch
from eval.evaluators import get_evaluator_class
from eval.eval import evaluate
from dataset.rolebench import get_evaluation_triplets
from configs.config import load_config

def parse_args():
    parser = argparse.ArgumentParser(description="Simple demo for evaluating generated images")
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing the images to evaluate"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--triplet",
        type=str,
        required=True,
        help="Triplet to evaluate against (e.g., 'mouse_chasing_cat')"
    )
    parser.add_argument(
        "--evaluator_model",
        type=str,
        default="vqascore_clip-flant5-xxl",
        help="Evaluator model to use"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./configs/config_sdxl_finetune_contrastive.ini",
        help="Path to config file"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config
    cfg = load_config(args.config_path)
    
    # Initialize evaluator
    print(f"Initializing evaluator: {args.evaluator_model}")
    evaluator = get_evaluator_class(
        args.evaluator_model,
        cfg.get("openai", {}).get("api_key") if "openai" in cfg else None
    )
    
    # Get evaluation triplets
    eval_triplets = get_evaluation_triplets(args.triplet)
    print(f"Will evaluate against triplets: {eval_triplets}")
    
    # Run evaluation for each evaluation triplet
    with torch.no_grad():
        for eval_triplet in eval_triplets:
            print(f"\nEvaluating against triplet: {eval_triplet}")
            
            results_dir = Path(args.output_dir) / eval_triplet / "results"
            os.makedirs(results_dir, exist_ok=True)
            
            try:
                # Note: Using metric="vqascore" explicitly
                qa_scores, qa_score_by_categories, image_QA_pairs = evaluate(
                    image_dir=args.image_dir,
                    eval_triplet=eval_triplet,
                    metric="vqascore",  # Using vqascore explicitly
                    evaluator=evaluator,
                    output_dir=results_dir,
                    cfg=cfg,
                    check_exists=False
                )
                
                # Print summary of results
                print("\nEvaluation Results:")
                print(f"Average QA Score: {sum(qa_scores.values()) / len(qa_scores):.3f}")
                print("\nScores by category:")
                for category, scores in qa_score_by_categories.items():
                    avg_score = sum(scores.values()) / len(scores)
                    print(f"{category}: {avg_score:.3f}")
                
            except Exception as e:
                print(f"Error evaluating triplet {eval_triplet}: {str(e)}")
                continue

if __name__ == "__main__":
    main()