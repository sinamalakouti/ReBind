import os
import csv
import pickle
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
from eval.eval import (
    get_normalized_relation_score,
    get_relation_entropy_scores,
    sg_check_VQA_train_prompt,
)
from modeling.utils.internVL_utils import get_model_internVL
from modeling.utils.llava_utils import get_model as get_model_llava


def get_MLLM_evaluator(MLLM_evaluator):
    if MLLM_evaluator == "llava":
        processor_eval, model_eval = get_model_llava(
            "llava-hf/llava-v1.6-vicuna-7b-hf", quantization=True
        )
    elif MLLM_evaluator == "internvl":
        processor_eval, model_eval = get_model_internVL(
            "OpenGVLab/InternVL2-8B", quantization=True
        )
    else:
        raise ValueError(f"model not registered! got: {MLLM_evaluator}")
    return processor_eval, model_eval


@torch.no_grad()
def llava_score(
    image,
    prompt_scene_graph,
    model_llava,
    processor_llava,
    device,
    quantization,
    MLLM_evaluator,
    return_QA_pairs=False,
):
    has_inconsistency, inconsistencies, queries, QA_pairs = sg_check_VQA_train_prompt(
        image,
        prompt_scene_graph,
        model_llava,
        processor_llava,
        device=device,
        quantization=quantization,
        MLLM_evaluator=MLLM_evaluator,
        num_questions=9,
    )
    num_inconsistency = len(inconsistencies)
    max_score = len(queries)
    assert (
        num_inconsistency <= max_score
    ), f"more than {max_score} inconsistency {num_inconsistency}"
    assert num_inconsistency >= 0, f"negative inconsistency {num_inconsistency}"
    score = (max_score - num_inconsistency) / max_score
    score = max(0.001, score)
    if return_QA_pairs:
        return score, QA_pairs
    else:
        return score


def get_next_image_number(log_image_dir):
    """Get the next available image number by checking existing files"""
    existing_numbers = set()
    for filename in os.listdir(log_image_dir):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            try:
                num = int(filename.split(".")[0])  # Get number from "1.png" etc
                existing_numbers.add(num)
            except ValueError:
                continue

    # Find the next available number
    next_num = 0
    while next_num in existing_numbers:
        next_num += 1
    return next_num


def eval_mllm(MLLM_evaluator_QA, log_dir, log_image_dir, rel_prompt_triplet, device):
    # Initialize models
    if MLLM_evaluator_QA == "llava":
        processor_eval, model_eval = get_model_llava(
            "llava-hf/llava-v1.6-vicuna-7b-hf", quantization=True
        )
    elif MLLM_evaluator_QA == "internvl":
        processor_eval, model_eval = get_model_internVL(
            "OpenGVLab/InternVL2-8B", quantization=True
        )
        processor_eval_llava, model_eval_llava = get_model_llava(
            "llava-hf/llava-v1.6-vicuna-7b-hf", quantization=True
        )
    else:
        raise ValueError(f"model not registered! got: {MLLM_evaluator_QA}")

    # Load existing metrics if they exist
    metrics_file = os.path.join(log_dir, f"metrics_{rel_prompt_triplet}.csv")
    existing_metrics = {}
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_metrics[int(row["image_num"])] = {
                    "qa_score": float(row["QA_score"]),
                    "rel_score": float(row["rel_score"]),
                    "total_entropy": float(row["entropy_score"]),
                    "true_rel_entropy": float(row["true_relation_entropy_score"]),
                }

    # Load existing QA pairs if they exist
    qa_file = os.path.join(log_dir, f"QA_pairs_{rel_prompt_triplet}.pkl")
    image_QA_pairs = {}
    if os.path.exists(qa_file):
        with open(qa_file, "rb") as f:
            image_QA_pairs = pickle.load(f)

    scores = []
    qa_scores = []
    rel_scores = []
    total_entropy_scores = []
    true_rel_entropy_scores = []
    image_numbers = []

    # Process each image
    for image_name in tqdm(sorted(os.listdir(log_image_dir))):
        if not image_name.endswith((".png", ".jpg", ".jpeg")):
            continue

        image_num = int(image_name.split(".")[0])

        # Skip if metrics already exist for this image
        if image_num in existing_metrics:
            metrics = existing_metrics[image_num]
            scores.append(metrics["qa_score"])
            qa_scores.append(metrics["qa_score"])
            rel_scores.append(metrics["rel_score"])
            total_entropy_scores.append(metrics["total_entropy"])
            true_rel_entropy_scores.append(metrics["true_rel_entropy"])
            image_numbers.append(image_num)
            continue

        image = Image.open(os.path.join(log_image_dir, image_name))

        # Calculate scores
        qa_score, QA_pairs = llava_score(
            image,
            rel_prompt_triplet,
            model_eval,
            processor_eval,
            device,
            quantization=True,
            MLLM_evaluator=MLLM_evaluator_QA,
            return_QA_pairs=True,
        )

        rel_score = get_normalized_relation_score(
            image,
            rel_prompt_triplet,
            model_eval if MLLM_evaluator_QA == "llava" else model_eval_llava,
            processor_eval if MLLM_evaluator_QA == "llava" else processor_eval_llava,
            device,
            quantization=True,
            MLLM_evaluator="llava",
        )

        entropy_score = get_relation_entropy_scores(
            image,
            rel_prompt_triplet,
            model_eval if MLLM_evaluator_QA == "llava" else model_eval_llava,
            processor_eval if MLLM_evaluator_QA == "llava" else processor_eval_llava,
            device,
            quantization=True,
            MLLM_evaluator="llava",
        )

        # Store results
        image_QA_pairs[str(image_num)] = QA_pairs
        score = qa_score
        scores.append(score)
        qa_scores.append(qa_score)
        rel_scores.append(rel_score)
        total_entropy_scores.append(entropy_score["total_entropy"])
        true_rel_entropy_scores.append(entropy_score["true_relation_entropy"])
        image_numbers.append(image_num)

    # Save QA pairs
    with open(qa_file, "wb") as f:
        pickle.dump(image_QA_pairs, f)

    # Save metrics
    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image_num",
                "QA_score",
                "rel_score",
                "entropy_score",
                "true_relation_entropy_score",
            ]
        )
        for idx, image_num in enumerate(image_numbers):
            writer.writerow(
                [
                    image_num,
                    qa_scores[idx],
                    rel_scores[idx],
                    total_entropy_scores[idx],
                    true_rel_entropy_scores[idx],
                ]
            )


# define vars
MLLM_evaluator_QA = "llava"
log_dir = "output/diffusion-dpo-softPrompt/INITPROMPTA photo of a mouse chasing a cat_NCTX8_EVALGOALcat_chasing_mouse_MLLMinternvl/final_results"
log_image_dir = os.path.join(log_dir, "images")
rel_prompt_triplets = ["cat_chasing_mouse", "mouse_chasing_cat"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Main execution
for rel_prompt_triplet in rel_prompt_triplets:
    eval_mllm(MLLM_evaluator_QA, log_dir, log_image_dir, rel_prompt_triplet, device)
