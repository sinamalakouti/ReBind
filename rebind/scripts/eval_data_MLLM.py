import argparse
import torch
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from modeling.utils.internVL_utils import get_model_internVL
from modeling.utils.llava_utils import get_model as get_model_llava
from eval.eval import (
    get_normalized_relation_score,
    get_relation_entropy_scores,
    sg_check_VQA_train_prompt,
)
import pickle


@torch.no_grad()
def llava_score(
    image,
    prompt_scene_graph,
    model_llava,
    processor_llava,
    device,
    quantization,
    args,
):
    has_inconsistency, inconsistencies, queries = sg_check_VQA_train_prompt(
        image,
        prompt_scene_graph,
        model_llava,
        processor_llava,
        device=device,
        quantization=quantization,
        MLLM_evaluator=args.MLLM_evaluator,
    )
    num_inconsistency = len(inconsistencies)
    max_score = len(queries)
    assert (
        num_inconsistency <= max_score
    ), f"more than {max_score} inconsistency {num_inconsistency}"
    assert num_inconsistency >= 0, f"negative inconsistency {num_inconsistency}"
    score = (max_score - num_inconsistency) / max_score
    score = max(0.001, score)
    return score


def evaluate_directory(data_path, prompt_scene_graph, args):
    """Evaluate all images in a directory using multiple metrics."""

    # Setup models
    if args.MLLM_evaluator == "llava":
        processor_eval, model_eval = get_model_llava(
            "llava-hf/llava-v1.6-vicuna-7b-hf", quantization=args.llava_quantization
        )
    elif args.MLLM_evaluator == "internvl":
        processor_eval, model_eval = get_model_internVL(
            "OpenGVLab/InternVL2-8B", quantization=args.llava_quantization
        )
    else:
        raise ValueError(f"model not registered! got: {args.MLLM_evaluator}")

    print("LLAAVA DEVICE IS: ", model_eval.device)
    with open(data_path, "rb") as f:
        data_tuples = pickle.load(f)
    # Get all images from directory
    images = [tup[0] for tup in data_tuples]

    # Initialize score lists
    qa_scores = []
    rel_scores = []
    total_entropies = []
    true_rel_entropies = []
    entropy_ratios = []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Evaluate each image
    for image in tqdm(images):
        image = image.convert("RGB")

        # Get QA score
        qa_score = llava_score(
            image,
            prompt_scene_graph,
            model_eval,
            processor_eval,
            device,
            quantization=args.llava_quantization,
            args=args,
        )

        # Get relation score
        rel_score = get_normalized_relation_score(
            image,
            prompt_scene_graph,
            model_eval,
            processor_eval,
            device,
            quantization=args.llava_quantization,
            MLLM_evaluator=args.MLLM_evaluator,
        )

        # Get entropy scores
        entropy_scores = get_relation_entropy_scores(
            image,
            prompt_scene_graph,
            model_eval,
            processor_eval,
            device,
            quantization=args.llava_quantization,
            MLLM_evaluator=args.MLLM_evaluator,
        )

        # Store scores
        qa_scores.append(qa_score)
        rel_scores.append(rel_score)
        total_entropies.append(entropy_scores["total_entropy"])
        true_rel_entropies.append(entropy_scores["true_relation_entropy"])
        entropy_ratios.append(
            (
                (entropy_scores["true_relation_entropy"] + 1e-10)
                / (entropy_scores["total_entropy"] + 1e-10)
            )
        )

    # Calculate statistics
    stats = {
        "Number of Images": len(images),
        "QA Scores": {
            "Mean": np.mean(qa_scores),
            "Std": np.std(qa_scores),
            "Min": np.min(qa_scores),
            "Max": np.max(qa_scores),
        },
        "Relation Scores": {
            "Mean": np.mean(rel_scores),
            "Std": np.std(rel_scores),
            "Min": np.min(rel_scores),
            "Max": np.max(rel_scores),
        },
        "Total Entropy": {
            "Mean": np.mean(total_entropies),
            "Std": np.std(total_entropies),
            "Min": np.min(total_entropies),
            "Max": np.max(total_entropies),
        },
        "True Relation Entropy": {
            "Mean": np.mean(true_rel_entropies),
            "Std": np.std(true_rel_entropies),
            "Min": np.min(true_rel_entropies),
            "Max": np.max(true_rel_entropies),
        },
        "Entropy Ratio": {
            "Mean": np.mean(entropy_ratios),
            "Std": np.std(entropy_ratios),
            "Min": np.min(entropy_ratios),
            "Max": np.max(entropy_ratios),
        },
    }

    # Print statistics
    logging.info("\nEvaluation Results:")
    for metric, values in stats.items():
        logging.info(f"\n{metric}:")
        if isinstance(values, dict):
            for stat, value in values.items():
                logging.info(f"{stat}: {value:.4f}")
        else:
            logging.info(f"{values}")

    return stats


if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--MLLM_evaluator", type=str, default="llava")
    parser.add_argument("--llava_quantization", type=bool, action="store_true")
    args = parser.parse_args()

    # Define scene graph for evaluation
    # prompt_scene_graph = [("mouse", "chasing", "cat")]
    prompt_scene_graph = "mouse_chasing_cat"

    # Run evaluation
    stats = evaluate_directory(args.data_path, prompt_scene_graph, args)
