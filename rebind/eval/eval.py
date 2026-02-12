import csv
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from dataset.paths import get_data_dir, get_qa_eval_json_path
from dataset.rolebench import (
    RELATIONS,
    get_evaluation_triplets,
    parse_action_triplet,
    reverse_triplet,
    rolebench_data,
)
from modeling.llm.models.gpt4 import GPT4
from modeling.utils.llava_utils import get_batch_response
from prompts.prompt_utils import (
    get_action_relation_question,
    get_base_prompt_from_action_triplet,
)

from prompts.constants import  RELATION_QUESTION_CATEGORIES
from prompts.templates.eval_templates import (
    create_llm_eval_template,
)

try:
    from eval.eval_utils import parse_yes_no
except ImportError:
    from eval_utils import parse_yes_no

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@torch.no_grad()
def sg_check_VQA_train_batch(
    images, pl_sg, model_llava, processor_llava, accelerator, quantization
):
    inconsistencies = []
    queries = {
        "object": "is there a {object} in this image? Answer yes or no.",
        "relation": "is {subject} {relation} {object} in this image? Answer yes or no.",
    }
    model_llava.to(accelerator.device) if not quantization else None
    for triplet in pl_sg:  # todo right now oly support one triplet
        subject, relation, object = triplet
        # check subject and object
        responses_subject = get_batch_response(
            model_llava,
            processor_llava,
            images,
            queries["object"].format(object=subject),
        )
        subject_responses_processed = [
            int(parse_yes_no(response) == "yes") for response in responses_subject
        ]
        subject_responses_processed = np.array(subject_responses_processed)

        responses_object = get_batch_response(
            model_llava,
            processor_llava,
            images,
            queries["object"].format(object=object),
        )
        object_responses_processed = [
            int(parse_yes_no(response) == "yes") for response in responses_object
        ]
        object_responses_processed = np.array(object_responses_processed)

        # check relation

        responses_relation = get_batch_response(
            model_llava,
            processor_llava,
            images,
            queries["relation"].format(
                subject=subject, relation=relation, object=object
            ),
        )
        relation_responses_processed = [
            int(parse_yes_no(response) == "yes") for response in responses_relation
        ]
        relation_responses_processed = np.array(relation_responses_processed)

    num_inconsistencies = (
        subject_responses_processed
        + object_responses_processed
        + relation_responses_processed
    )
    return num_inconsistencies


def _get_chat_response(
    model=None,
    processor=None,
    image=None,
    question=None,
    MLLM_evaluator=None,
    max_new_tokens=5,
    temperature=0,
):
    assert MLLM_evaluator is not None, "MLLM_evaluator must be provided"
    assert image is not None, "image must be provided"
    assert question is not None, "question must be provided"

    if MLLM_evaluator == "llava":
        from modeling.utils.llava_utils import get_chat_response

        assert (
            model is not None and processor is not None
        ), "model and processor must be provided for llava"

        return get_chat_response(
            model,
            processor,
            image,
            question,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    elif MLLM_evaluator == "internvl":  # Assuming InternVL uses AutoModel'
        from modeling.utils.internVL_utils import get_chat_response

        assert (
            model is not None and processor is not None
        ), "model and processor must be provided for internvl"

        return get_chat_response(
            model,
            processor,
            image,
            question,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    elif MLLM_evaluator == "gpt4v":
        from modeling.utils.gpt4v_utils import get_chat_response

        return get_chat_response(
            image, question, max_new_tokens=max_new_tokens, temperature=temperature
        )
    else:
        raise ValueError(f"model {model} is not registered")


def _get_options_logprobs(
    model, processor, image, question, n_tokens, MLLM_evaluator, options=None
):
    if MLLM_evaluator == "llava":
        from modeling.utils.llava_utils import get_options_logprobs

        return get_options_logprobs(
            model, processor, image, question, options, n_tokens
        )
    elif MLLM_evaluator == "internvl":
        from modeling.utils.internVL_utils import get_options_logprobs

        return get_options_logprobs(
            model, processor, image, question, options, n_tokens
        )
    else:
        raise ValueError(f"model {model} is not registered")




@torch.no_grad()
def compute_qa_score(
    image,
    triplet,
    prompt,
    model,
    processor,
    device,
    quantization,
    MLLM_evaluator,
):
    inconsistencies = []
    eval_json_path = get_qa_eval_json_path(triplet, cfg["model"]["llm_model"])
    logging.info(f"FOUND EVAL_JSON_PATH: {eval_json_path}")
    if not eval_json_path.exists():
        os.makedirs(os.path.dirname(), exist_ok=True)
        qa = generate_qa_eval_json(prompt)
        try:
            with open(eval_json_path, "w") as f:
                json.dump(qa, f, indent=2)
            logger.info(f"Successfully saved evaluations to {eval_json_path}")
        except Exception as e:
            logger.error(f"Error saving evaluations to file: {str(e)}")

    with open(eval_json_path, "r") as f:
        questions_by_category = json.load(f)
    QA_by_category = {}
    model.to(device) if not quantization and model is not None else None
    correct_answer = 0
    print("questions_by_category")
    print(questions_by_category)
    for category in questions_by_category.keys():
        QA_by_category[category] = {
            "weight": questions_by_category[category]["weight"],
            "questions": {},
        }
        for question in questions_by_category[category]["questions"]:
            response = _get_chat_response(
                model, processor, image, question, MLLM_evaluator
            )
            response_processed = parse_yes_no(response) == "yes"
            QA_by_category[category]["questions"][question] = (
                "yes" if response_processed else "no"
            )
            if not response_processed:

                print("question: ", question)
                print("response: ", response)

            if response_processed:
                correct_answer += 1
            else:
                inconsistencies.append(question)
    print("QA_by_category")
    print(QA_by_category)
    return (
        len(inconsistencies) > 0,
        inconsistencies,
        QA_by_category,
    )


@torch.no_grad()
def get_mllm_QA_score_from_json(
    image,
    triplet,
    prompt,
    model,
    processor,
    device,
    quantization,
    MLLM_evaluator,
    return_QA_pairs=True,
):
    has_inconsistency, inconsistencies, QA_by_category = compute_qa_score(
        image,
        triplet,
        prompt,
        model,
        processor,
        device=device,
        quantization=quantization,
        MLLM_evaluator=MLLM_evaluator,
    )
    num_inconsistency = len(inconsistencies)
    max_score = sum(len(questions) for questions in QA_by_category.values())
    obj_det_score_coeff = 1
    for questions in QA_by_category["detection"]:
        if QA_by_category["detection"][questions] == "no":
            obj_det_score_coeff = 0
    assert (
        num_inconsistency <= max_score
    ), f"more than {max_score} inconsistency {num_inconsistency}"
    assert num_inconsistency >= 0, f"negative inconsistency {num_inconsistency}"
    score = (max_score - num_inconsistency) / max_score
    print("image: ", image)
    print("num_inconsistency: ", num_inconsistency)
    print("max_score: ", max_score)
    score = max(0.001, score * obj_det_score_coeff)
    if return_QA_pairs:
        return score, QA_by_category
    else:
        return score


@torch.no_grad()
def get_normalized_relation_score(
    image,
    action_relation_triplet,
    model,
    processor,
    device,
    quantization,
    MLLM_evaluator,
):
    relations = RELATIONS
    subject, GT_relation, obj = parse_action_triplet(action_relation_triplet)
    assert (
        GT_relation in RELATIONS
    ), f"MLLM_eval_prompts.py: true relation {GT_relation} not in relations"
    # Constants
    REGULARIZATION_VALUE = -1e10
    EPSILON = 1e-10

    model.to(device) if not quantization else None

    # Construct question with explicit options
    question = get_action_relation_question(action_relation_triplet)

    # Get logprobs for all possible relations
    results = _get_options_logprobs(
        model=model,
        processor=processor,
        image=image,
        question=question,
        options=RELATIONS,
        n_tokens=5,
        MLLM_evaluator=MLLM_evaluator,
    )

    # Extract normalized log probabilities
    logprobs = results["scores"]
    all_log_probs = torch.tensor(
        [logprobs[relation]["log_prob"] for relation in relations]
    )

    # Regularize -inf values
    all_log_probs_regularized = torch.where(
        all_log_probs == float("-inf"),
        torch.tensor(REGULARIZATION_VALUE),
        all_log_probs,
    )

    # Get the log probability of the correct relation
    logprob_correct = logprobs[GT_relation]["log_prob"]

    if torch.all(all_log_probs_regularized == REGULARIZATION_VALUE):
        return 0.0

    if logprob_correct == float("-inf"):
        return 0.0

    # Compute Log-Sum-Exp for all relations
    lse_all = torch.logsumexp(all_log_probs_regularized, dim=-1)

    # Compute Log-Score for the correct relation
    log_score = logprob_correct - lse_all

    # Find actual max log score from the data
    log_score_max = torch.max(all_log_probs_regularized) - lse_all
    log_score_min = torch.min(all_log_probs_regularized) - lse_all

    # Avoid division by zero
    denominator = log_score_max - log_score_min
    if abs(denominator) < EPSILON:
        return 1.0 if log_score == log_score_max else 0.0

    normalized_score = (log_score - log_score_min) / denominator
    print("log_score: ", log_score)
    print("log_score_min: ", log_score_min)
    print("log_score_max: ", log_score_max)
    print("denominator: ", denominator)
    print("normalized_score: ", normalized_score)
    return normalized_score.item()


@torch.no_grad()
def get_relation_entropy_scores(
    image, relation, model, processor, device, quantization, MLLM_evaluator
):

    subject, true_relation, obj = parse_action_triplet(relation)
    assert (
        true_relation in RELATIONS
    ), f"get_relation_entropy_scores: true relation {true_relation} not in relations"

    # Constants
    REGULARIZATION_VALUE = -1e10
    EPSILON = 1e-10

    # model.to(device) if not quantization else None

    # Construct question with explicit options
    question = get_action_relation_question(relation)

    # Get logprobs for all possible relations
    results = _get_options_logprobs(
        model=model,
        processor=processor,
        image=image,
        question=question,
        options=RELATIONS,
        n_tokens=5,
        MLLM_evaluator=MLLM_evaluator,
    )

    # Extract log probabilities
    logprobs = results["scores"]
    all_log_probs = torch.tensor(
        [logprobs[relation]["log_prob"] for relation in RELATIONS]
    )

    # Regularize -inf values
    all_log_probs_regularized = torch.where(
        all_log_probs == float("-inf"),
        torch.tensor(REGULARIZATION_VALUE),
        all_log_probs,
    )

    # Convert log probabilities to probabilities
    # First subtract max for numerical stability
    log_probs_shifted = all_log_probs_regularized - torch.max(all_log_probs_regularized)
    probs = torch.exp(log_probs_shifted)

    # Normalize to get proper probability distribution
    probs = probs / (probs.sum() + EPSILON)

    # Calculate total entropy: -sum(p * log(p))
    total_entropy = -torch.sum(probs * torch.log(probs + EPSILON))

    # Get probability of true relation
    true_rel_idx = list(RELATIONS).index(true_relation)
    true_rel_prob = probs[true_rel_idx]

    # Calculate entropy contribution of true relation: -p * log(p)
    true_rel_entropy = -true_rel_prob * torch.log(true_rel_prob + EPSILON)

    return {
        "total_entropy": total_entropy.item(),
        "true_relation_entropy": true_rel_entropy.item(),
        "true_relation_prob": true_rel_prob.item(),
    }


@torch.no_grad()
def _get_options_VQA_scores(model, images, relation_triplet):
    from prompts.templates.eval_templates import RELATION_PROMPT_TEMPLATE

    subject, true_relation, obj = parse_action_triplet(relation_triplet)
    questions = []
    for relation in RELATIONS:
        questions.append(
            RELATION_PROMPT_TEMPLATE[relation].format(subject=subject, object=obj)
        )
    scores = model(
        images=images, texts=questions
    )  # scores[i][j] is the score between image i

    return scores


@torch.no_grad()
@torch.inference_mode()
def get_relation_VQA_scores(model, images, relation_triplet):
    # Get raw probabilities for each relation
    scores = _get_options_VQA_scores(
        model, images, relation_triplet
    )  # [batch, num_relations]
    scores = scores.squeeze(0)
    scores = torch.clamp(scores, min=0.001)  # Ensure no zeros
    # Get the true relation's index
    _, true_relation, _ = parse_action_triplet(relation_triplet)
    true_rel_idx = list(RELATIONS).index(true_relation)

    # Get normalized score with numerical stability
    raw_true_score = scores[true_rel_idx].item()  # Using [0] since batch_size=1

    total_score = torch.sum(scores)

    normalized_score = raw_true_score / total_score.item()

    return {"normalized_score": normalized_score, "true_score": raw_true_score}


@torch.no_grad()
@torch.inference_mode()
def get_relation_entropy_VQA_scores(model, images, relation_triplet):
    # Get true relation index
    _, true_relation, _ = parse_action_triplet(relation_triplet)
    true_rel_idx = list(RELATIONS).index(true_relation)

    # Get raw probabilities for each relation
    scores = _get_options_VQA_scores(model, images, relation_triplet)
    scores = scores.squeeze(0)
    scores = torch.clamp(scores, min=0.001)  # Ensure no zeros

    scores = scores / scores.sum()  # normalize scores;

    # Calculate entropy scores with numerical stability
    entropy_scores = -scores * torch.log(scores)
    total_entropy = torch.sum(entropy_scores)
    true_rel_prob = scores[true_rel_idx]
    true_rel_entropy = entropy_scores[true_rel_idx]
    assert (
        entropy(true_rel_prob) == true_rel_entropy
    ), "entropy of true relation is not correct"
    return {
        "total_entropy": total_entropy.item(),
        "true_relation_entropy": true_rel_entropy.item(),
        "normalized_total_entropy": true_rel_prob.item(),
    }


def entropy(score):
    return -score * torch.log(score)


def evaluate_qa_score(
    image_dir,
    eval_triplet,
    evaluator_model,
    output_dir,
    qa_eval_json_path,
    check_exists=False,
):
    output_dir = Path(output_dir) / "qa_score" / evaluator_model.name
    os.makedirs(output_dir, exist_ok=True)

    QA_pairs_path = output_dir / f"qa_pairs_{eval_triplet}.json"
    csv_path = (
        output_dir / f"qa_scores_{evaluator_model.name}_triplet_{eval_triplet}.csv"
    )

    # Check if results already exist
    if check_exists and csv_path.exists() and QA_pairs_path.exists():
        logger.info(f"Found existing results at {csv_path}")
        try:
            # Load existing results
            with open(QA_pairs_path, "r") as f:
                image_QA_pairs = json.load(f)

            # Read CSV for scores
            df = pd.read_csv(csv_path)
            qa_scores = dict(zip(df["image_num"], df["total"]))

            # Extract category scores
            categories = RELATION_QUESTION_CATEGORIES
            qa_score_by_categories = {}
            for idx, row in df.iterrows():
                image_name = row["image_num"]
                qa_score_by_categories[image_name] = {
                    category: row[category] for category in categories
                }

            return qa_scores, qa_score_by_categories, image_QA_pairs
        except Exception as e:
            logger.warning(f"Error loading existing results, will recompute: {e}")

    # If we get here, either check_exists=False or loading failed
    image_nums = []
    qa_scores = {}
    image_QA_pairs = {}
    qa_score_by_categories = {}
    categories = RELATION_QUESTION_CATEGORIES

    # Get all categories from RELATION_QUESTION_CATEGORIES
    categories = RELATION_QUESTION_CATEGORIES

    for image_name in tqdm(os.listdir(image_dir)):
        image_num = image_name.split(".")[0]
        image_nums.append(image_num)
        image_path = os.path.join(image_dir, image_name)
        qa_score, qa_score_by_category, qa_by_category = evaluator_model.qa_score(
            image_path,
            eval_triplet,
            categories,
            mode="weighted_average",
            qa_eval_json_path=qa_eval_json_path,
        )

        qa_scores[image_name] = qa_score
        qa_score_by_categories[image_name] = qa_score_by_category
        image_QA_pairs[image_name] = qa_by_category

    with open(QA_pairs_path, "w") as f:
        json.dump(image_QA_pairs, f, indent=2)

    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        # Write header with total and each category
        header = ["image_num", "total"] + categories
        writer.writerow(header)

        # Write scores for each image
        for image_name in image_QA_pairs.keys():
            row = [image_name, qa_scores[image_name]]
            # Add score for each category
            for category in categories:
                row.append(qa_score_by_categories[image_name][category])
            writer.writerow(row)
    return qa_scores, qa_score_by_categories, image_QA_pairs


def evaluate_VQAscore(
    image_dir, eval_triplet, evaluator_model, output_dir, check_exists=False
):
    output_dir = Path(output_dir) / "VQAscore" / evaluator_model.name
    os.makedirs(output_dir, exist_ok=True)

    csv_path = (
        output_dir / f"VQA_scores_{evaluator_model.name}_triplet_{eval_triplet}.csv"
    )

    # Check if results already exist
    if check_exists and csv_path.exists():
        logger.info(f"Found existing VQA results at {csv_path}")
        try:
            # Load existing results
            df = pd.read_csv(csv_path)
            VQAScores = dict(zip(df["image_num"], df["total"]))
            return VQAScores, None, None
        except Exception as e:
            logger.warning(f"Error loading existing VQA results, will recompute: {e}")

    # If we get here, either check_exists=False or loading failed
    image_nums = []
    VQAScores = {}

    for image_name in tqdm(os.listdir(image_dir)):
        image_num = image_name.split(".")[0]
        image_nums.append(image_num)
        image_path = os.path.join(image_dir, image_name)
        VQAScore = evaluator_model.get_VQA_score(image_path, eval_triplet)
        VQAScores[image_name] = VQAScore

    # Save results
    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        header = ["image_num", "total"]
        writer.writerow(header)
        for image_name in VQAScores.keys():
            row = [image_name, VQAScores[image_name]]
            writer.writerow(row)

    return VQAScores, None, None


def generate_qa_eval_json(prompt):
    QA_gen_prompt = create_llm_eval_template()
    LLM = GPT4(model="gpt-4-turbo")
    print(QA_gen_prompt)
    evaluation_response = LLM(
        system_prompt=QA_gen_prompt["system_prompt"],
        user_prompt=QA_gen_prompt["user_prompt"].format(prompt=prompt),
    )

    print(f"EVALUATION RESPONSE: {evaluation_response}")

    # Parse response and store with original structure
    evaluation_data = json.loads(evaluation_response)

    return evaluation_data


def evaluate(
    image_dir, eval_triplet, metric, evaluator, output_dir, cfg, check_exists=False
):
    print(f"EVALUATING {metric} for triplet {eval_triplet}")

    if metric == "qa_score":
        qa_eval_json_path = get_qa_eval_json_path(
            eval_triplet, cfg["model"]["llm_model"]
        )
        if not qa_eval_json_path.exists():
            base_prompt = get_base_prompt_from_action_triplet(eval_triplet)
            qa = generate_qa_eval_json(base_prompt)
            try:
                with open(qa_eval_json_path, "w") as f:
                    json.dump(qa, f, indent=2)
                logger.info(f"Successfully saved evaluations to {qa_eval_json_path}")
            except Exception as e:
                logger.error(f"Error saving evaluations to file: {str(e)}")

        return evaluate_qa_score(
            image_dir,
            eval_triplet,
            evaluator,
            output_dir,
            qa_eval_json_path,
            check_exists,
        )
    elif metric == "rel_score":
        pass
        # evaluate_relation_score(image_dir, triplet, evaluator_model, output_dir)
    elif metric == "entropy_score":
        pass
        # evaluate_entropy_score(image_dir, triplet, evaluator_model, output_dir)
    elif metric == "vqascore":
        return evaluate_VQAscore(
            image_dir, eval_triplet, evaluator, output_dir, check_exists
        )
    else:
        raise ValueError(f"Metric {metric} not supported")


def print_summary(output_dir, evaluator_name, gen_triplet=None, eval_triplet=None):
    """Print summary of evaluation metrics from result files.

    Args:
        output_dir (str): Base output directory
        evaluator_name (str): Name of the evaluator (e.g., 'internvl', 'llava', 'gpt4v')
        gen_triplet (str, optional): Generation triplet used in path
        eval_triplet (str): Evaluation triplet used in file names
    """
    output_dir = Path(output_dir)
    metrics_summary = {}

    # Common metric paths and their file patterns
    metric_patterns = {
        "qa_score": {
            "subpath": f"qa_score/{evaluator_name}",
            "file_pattern": f"qa_scores_{evaluator_name}_triplet_{eval_triplet}.csv",
        },
        "rel_score": {
            "subpath": f"rel_score/{evaluator_name}",
            "file_pattern": f"rel_scores_{evaluator_name}_triplet_{eval_triplet}.csv",
        },
        "entropy_score": {
            "subpath": f"entropy_score/{evaluator_name}",
            "file_pattern": f"entropy_scores_{evaluator_name}_triplet_{eval_triplet}.csv",
        },
    }

    # Look for results in both possible directory structures
    possible_result_dirs = []
    if gen_triplet:
        # Structure 1: data/chasing/mouse_chasing_cat/dall-e-3/expanded_results/...
        possible_result_dirs.append(output_dir / "expanded_results")
        # Structure 2: output/mouse_chasing_cat/validation/epoch_1/.../results/...
        possible_result_dirs.append(output_dir / "results")
    else:
        possible_result_dirs.append(output_dir)

    logger.info(f"Searching for metric files for evaluator: {evaluator_name}")

    # Search for metric files
    for metric, pattern_info in metric_patterns.items():
        for result_dir in possible_result_dirs:
            metric_dir = result_dir / pattern_info["subpath"]
            if not metric_dir.exists():
                continue

            metric_file = metric_dir / pattern_info["file_pattern"]
            if metric_file.exists():
                logger.info(f"Found {metric} file: {metric_file}")

                # Read and process CSV file
                try:
                    df = pd.read_csv(metric_file)
                    metrics_summary[metric] = {
                        "mean": df["total"].mean(),
                        "min": df["total"].min(),
                        "max": df["total"].max(),
                        "std": df["total"].std(),
                        # Add category scores if they exist
                        "categories": {
                            col: df[col].mean()
                            for col in df.columns
                            if col not in ["image_num", "total"]
                        },
                    }
                except Exception as e:
                    logger.error(f"Error reading {metric} file: {e}")
                break  # Found and processed the file, move to next metric

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info(f"Evaluation Summary for {evaluator_name}:")
    for metric, stats in metrics_summary.items():
        logger.info(f"\n{metric.upper()}:")
        logger.info(f"  Mean: {stats['mean']:.3f}")
        logger.info(f"  Min:  {stats['min']:.3f}")
        logger.info(f"  Max:  {stats['max']:.3f}")
        logger.info(f"  Std:  {stats['std']:.3f}")

        if stats["categories"]:
            logger.info("  Category Scores:")
            for category, score in stats["categories"].items():
                logger.info(f"    {category}: {score:.3f}")

    logger.info("=" * 50 + "\n")

    return metrics_summary


def print_summary_with_delta(output_dir, evaluator_name, gen_triplet=None):
    """Print summary of evaluation metrics with deltas between matching and unmatching triplets."""

    output_dir = Path(output_dir)
    metrics_summary = {}

    # Get matching and unmatching triplets
    eval_triplets = get_evaluation_triplets(gen_triplet)
    matching_triplet, unmatching_triplet = eval_triplets

    metric_patterns = {
        "qa_score": {
            "subpath": f"qa_score/{evaluator_name}",
            "file_pattern": lambda t: f"qa_scores_{evaluator_name}_triplet_{t}.csv",
        },
        "rel_score": {
            "subpath": f"rel_score/{evaluator_name}",
            "file_pattern": lambda t: f"rel_scores_{evaluator_name}_triplet_{t}.csv",
        },
        "entropy_score": {
            "subpath": f"entropy_score/{evaluator_name}",
            "file_pattern": lambda t: f"entropy_scores_{evaluator_name}_triplet_{t}.csv",
        },
    }

    # Look for results in both possible directory structures
    possible_result_dirs = []
    if "sdxl_lora" in str(output_dir):
        # For fine-tuned models, results are directly in the results directory
        possible_result_dirs.append(output_dir)
    else:
        # For pretrained models, check expanded_results
        possible_result_dirs.append(output_dir / "expanded_results")

    logger.info(f"Computing deltas for evaluator: {evaluator_name}")

    # Process each metric
    for metric, pattern_info in metric_patterns.items():
        metrics_summary[metric] = {"matching": None, "unmatching": None, "delta": None}

        for triplet_type, eval_triplet in [
            ("matching", matching_triplet),
            ("unmatching", unmatching_triplet),
        ]:
            for result_dir in possible_result_dirs:
                metric_dir = result_dir / pattern_info["subpath"]
                if not metric_dir.exists():
                    logger.warning("metric_dir does not exist: ", metric_dir)
                    continue

                metric_file = metric_dir / pattern_info["file_pattern"](eval_triplet)
                if metric_file.exists():
                    logger.info(
                        f"Found {metric} file for {triplet_type}: {metric_file}"
                    )

                    try:
                        df = pd.read_csv(metric_file)
                        stats = {
                            "categories": {
                                col: df[col].mean()
                                for col in df.columns
                                if col not in ["image_num", "total"]
                            },
                            "mean": df["total"].mean(),
                        }
                        metrics_summary[metric][triplet_type] = stats
                    except Exception as e:
                        logger.error(
                            f"Error reading {metric} file for {triplet_type}: {e}, path: {metric_file}"
                        )
                    break

        # Compute deltas if both matching and unmatching exist
        if (
            metrics_summary[metric]["matching"] is not None
            and metrics_summary[metric]["unmatching"] is not None
        ):

            metrics_summary[metric]["delta"] = {
                "mean": metrics_summary[metric]["matching"]["mean"]
                - metrics_summary[metric]["unmatching"]["mean"],
                "categories": {
                    cat: metrics_summary[metric]["matching"]["categories"][cat]
                    - metrics_summary[metric]["unmatching"]["categories"][cat]
                    for cat in metrics_summary[metric]["matching"]["categories"].keys()
                },
            }

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info(f"Evaluation Summary for {evaluator_name}:")
    for metric, results in metrics_summary.items():
        if results["matching"] is None or results["unmatching"] is None:
            logger.info(f"\n{metric.upper()}: No data found")
            continue

        logger.info(f"\n{metric.upper()}:")
        logger.info(f"  Matching Mean:     {results['matching']['mean']:.3f}")
        logger.info(f"  Unmatching Mean:   {results['unmatching']['mean']:.3f}")
        logger.info(f"  Delta Mean:        {results['delta']['mean']:.3f}")

        if results["matching"]["categories"]:
            logger.info("  Category Deltas:")
            for category in results["matching"]["categories"].keys():
                logger.info(f"    {category}:")
                logger.info(
                    f"      Matching:   {results['matching']['categories'][category]:.3f}"
                )
                logger.info(
                    f"      Unmatching: {results['unmatching']['categories'][category]:.3f}"
                )
                logger.info(
                    f"      Delta:      {results['delta']['categories'][category]:.3f}"
                )

    logger.info("=" * 50 + "\n")

    return metrics_summary


def analyze_all_results(model_name, evaluator_name, output_file):
    """Analyze results for all triplets in rolebench_data and save to file.

    Args:
        model_name (str): Name of the generation model (e.g., 'dall-e-3')
        evaluator_name (str): Name of the evaluator (e.g., 'internvl', 'llava')
        output_file (str): Path to save the results
    """
    # Set up file logging
    file_handler = logging.FileHandler(output_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(message)s"))

    # Add file handler to logger
    logger.addHandler(file_handler)

    # Write header
    logger.info(f"Analysis Results")
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Evaluator: {evaluator_name}")
    logger.info(f"{'='*80}\n")

    all_metrics = {}

    for relation in RELATIONS:
        logger.info(f"\n{'-'*40}")
        logger.info(f"Relation: {relation}")
        logger.info(f"{'-'*40}")

        all_metrics[relation] = {}
        for triplet_type in ["frequent", "rare"]:
            triplet = rolebench_data[relation][triplet_type]

            logger.info(f"\nTriplet ({triplet_type}): {triplet}")

            # Get the paths
            triplet_dir, prompt_dir, image_dir = get_data_dir(
                triplet, model_name, LLM="gpt-4o"
            )

            # Results directory
            output_dir = triplet_dir / model_name

            # Get summary with deltas
            metrics = print_summary_with_delta(
                output_dir=output_dir,
                evaluator_name=evaluator_name,
                gen_triplet=triplet,
            )

            all_metrics[relation][triplet_type] = metrics

    logger.info(f"\n{'='*80}")
    logger.info("Analysis complete!")
    logger.info(f"{'='*80}")

    # Remove file handler
    logger.removeHandler(file_handler)

    return all_metrics


def create_comparison_csv(
    model_names,
    evaluator_names,
    categories,
    output_file="comparison_results_VQAScore_gpt-4o.csv",
):
    """Create a CSV comparing QA scores across models and evaluators and an averaged CSV.

    Args:
        model_names (list): List of T2I model names (e.g., ['dall-e-3', 'sdxl'])
        evaluator_names (list): List of evaluator names (e.g., ['llava', 'internvl'])
        categories (list): List of categories to evaluate (e.g., ['detection', 'spatial'])
        output_file (str): Path to save the CSV file
    """
    import csv
    from collections import defaultdict

    # Initialize the data structure for CSV
    rows = []

    # Create header rows
    header1 = ["Generation Triplet", "T2I"]
    header2 = ["", ""]  # Second row for subcategories

    # Add QAScore columns for each evaluator
    for evaluator in evaluator_names:
        # First row: evaluator name spanning all categories + total
        header1.extend([f"QAScore ({evaluator})"] * (len(categories) * 3 + 3))
        # Second row: category names with matching/unmatching/delta for each
        for category in categories:
            header2.extend(
                [f"{category}_matching", f"{category}_unmatching", f"{category}_delta"]
            )
        header2.extend(["total_matching", "total_unmatching", "total_delta"])

    rows.append(header1)
    rows.append(header2)

    # Dictionary to accumulate scores for averaging
    averages = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )
    )

    # Process each relation and triplet
    for relation in RELATIONS:
        for triplet_type in ["frequent", "rare"]:
            triplet = rolebench_data[relation][triplet_type]

            # Process each model
            for model_name in model_names:
                triplet_dir, _, _ = get_data_dir(triplet, model_name, LLM="gpt-4o")
                output_dir = triplet_dir / model_name

                row = [triplet, model_name]

                # Get scores for each evaluator
                for evaluator_name in evaluator_names:
                    metrics = print_summary_with_delta(
                        output_dir=output_dir,
                        evaluator_name=evaluator_name,
                        gen_triplet=triplet,
                    )

                    # Add scores for each category
                    for category in categories:
                        if (
                            metrics["qa_score"]["matching"]
                            and metrics["qa_score"]["unmatching"]
                        ):
                            matching = metrics["qa_score"]["matching"]["categories"][
                                category
                            ]
                            unmatching = metrics["qa_score"]["unmatching"][
                                "categories"
                            ][category]
                            delta = matching - unmatching
                            row.extend(
                                [f"{matching:.3f}", f"{unmatching:.3f}", f"{delta:.3f}"]
                            )
                            # Accumulate scores for averaging
                            averages[triplet][evaluator_name][category].append(
                                (matching, unmatching, delta)
                            )
                        else:
                            row.extend(["N/A", "N/A", "N/A"])

                    # Add total scores
                    if (
                        metrics["qa_score"]["matching"]
                        and metrics["qa_score"]["unmatching"]
                    ):
                        total_matching = metrics["qa_score"]["matching"]["mean"]
                        total_unmatching = metrics["qa_score"]["unmatching"]["mean"]
                        total_delta = metrics["qa_score"]["delta"]["mean"]
                        row.extend(
                            [
                                f"{total_matching:.3f}",
                                f"{total_unmatching:.3f}",
                                f"{total_delta:.3f}",
                            ]
                        )
                        # Accumulate total scores for averaging
                        averages[triplet][evaluator_name]["total"].append(
                            (total_matching, total_unmatching, total_delta)
                        )
                    else:
                        row.extend(["N/A", "N/A", "N/A"])

                rows.append(row)

    # Write to CSV
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    logger.info(f"Comparison results saved to {output_file}")

    # Calculate averages and write to a new CSV
    avg_rows = [header1, header2]
    for triplet in averages:
        avg_row = [triplet, "Average"]
        for evaluator_name in evaluator_names:
            for category in categories:
                if averages[triplet][evaluator_name][category]:
                    matching_avg = sum(
                        x[0] for x in averages[triplet][evaluator_name][category]
                    ) / len(averages[triplet][evaluator_name][category])
                    unmatching_avg = sum(
                        x[1] for x in averages[triplet][evaluator_name][category]
                    ) / len(averages[triplet][evaluator_name][category])
                    delta_avg = sum(
                        x[2] for x in averages[triplet][evaluator_name][category]
                    ) / len(averages[triplet][evaluator_name][category])
                    avg_row.extend(
                        [
                            f"{matching_avg:.3f}",
                            f"{unmatching_avg:.3f}",
                            f"{delta_avg:.3f}",
                        ]
                    )
                else:
                    avg_row.extend(["N/A", "N/A", "N/A"])
            if averages[triplet][evaluator_name]["total"]:
                total_matching_avg = sum(
                    x[0] for x in averages[triplet][evaluator_name]["total"]
                ) / len(averages[triplet][evaluator_name]["total"])
                total_unmatching_avg = sum(
                    x[1] for x in averages[triplet][evaluator_name]["total"]
                ) / len(averages[triplet][evaluator_name]["total"])
                total_delta_avg = sum(
                    x[2] for x in averages[triplet][evaluator_name]["total"]
                ) / len(averages[triplet][evaluator_name]["total"])
                avg_row.extend(
                    [
                        f"{total_matching_avg:.3f}",
                        f"{total_unmatching_avg:.3f}",
                        f"{total_delta_avg:.3f}",
                    ]
                )
            else:
                avg_row.extend(["N/A", "N/A", "N/A"])
        avg_rows.append(avg_row)

    avg_output_file = f"avg_{output_file}"
    with open(avg_output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(avg_rows)

    logger.info(f"Averaged comparison results saved to {avg_output_file}")


def get_latest_epoch_dir(base_path):
    """Find the latest epoch directory by number"""
    epoch_dirs = [d for d in base_path.glob("epoch_*") if d.is_dir()]
    if not epoch_dirs:
        return None

    # Extract numbers and find max
    latest_epoch = max(epoch_dirs, key=lambda x: int(str(x.name).split("_")[1]))
    return latest_epoch


def get_model_name(
    base_model, is_pretrained, is_expanded, training_mode=None, use_freq_rare=False
):
    if "rrnet" in base_model.lower():
        return "rrnet"
    if "sld" in base_model:
        return base_model
    elif "rpg" in base_model.lower():
        return "rpg"
    elif is_pretrained:
        return f"{base_model}_{'expanded' if is_expanded else 'basic'}"
    else:
        return f"{base_model}_{training_mode}_{'intermediate' if not use_freq_rare else 'use_freq_rare'}"


def get_model_validation_path(
    base_dir,
    model,
    evaluator_name,
    triplet,
    eval_triplet,
    metric,
    is_expanded,
    is_pretrained,
    use_freq_rare=False,
    training_mode=None,
):
    metric = get_metric_name(metric)
    _, relation, _ = parse_action_triplet(triplet)
    if "rpg" in model.lower():
        validation_dir = (
            Path(base_dir)
            / "ICCV_output"
            / "RPG"
            / triplet
            / "validation"
            / triplet
            # / "results"
            / (metric if metric != "VQAScore" else "VQAscore")
            / evaluator_name
        )
    elif "r2f" in model.lower():
        validation_dir = (
            Path(base_dir)
            / "ICCV_output"
            / "R2F"
            / "sdxl"
            / "validation"
            / "results"
            / triplet
            / (metric if metric != "VQAScore" else "VQAscore")
            / evaluator_name
        )
    elif "rrnet" in model.lower():
        validation_dir = (
            Path(base_dir)
            / "ICCV_output"
            / "RRNET"
            / "w=0.6"
            / triplet
            / "validation"
            / triplet
            / (metric if metric != "VQAScore" else "VQAscore")
            / evaluator_name
        )
    elif "sld" in model:
        validation_dir = (
            Path(base_dir)
            / "outputs"
            / f"{model}_{triplet}_{triplet}"
            / "sld_baseline"
            / f"{triplet}_to_{triplet}"
            / model.split("_")[1]
            / "validation"
            / eval_triplet
            / "results"
            / (metric if metric != "VQAScore" else "VQAscore")
            / evaluator_name
        )
    elif not is_pretrained:
        tmp = "use_freq_rare" if use_freq_rare else "intermediate"
        validation_dir = (
            Path(base_dir)
            / "ICCV_output"
            / model
            / training_mode
            / rolebench_data[relation]["rare"]
            / tmp
            / "validation"
        )
        epoch_dir = get_latest_epoch_dir(validation_dir)
        validation_dir = (
            validation_dir
            / epoch_dir
            / triplet
            / ("results" if not is_expanded else "expanded_results")
            / (metric if metric != "VQAScore" else "VQAscore")
            / evaluator_name
        )
    else:
        _, relation, _ = parse_action_triplet(triplet)
        validation_dir = (
            Path(base_dir)
            / "data"
            / relation
            / triplet
            / model
            / ("results" if not is_expanded else "expanded_results")
            / (metric if metric.lower() != "vqascore" else "VQAscore")
            / evaluator_name
        )
    return validation_dir



def get_model_images_dir(
    base_dir,
    model,
    triplet,
    is_expanded,
    is_pretrained,
    use_freq_rare=False,
    training_mode=None,
):
    _, relation, _ = parse_action_triplet(triplet)
    if "rpg" in model.lower():
        images_dir = (
            Path(base_dir) / "ICCV_output" / "RPG" / "sdxl" / triplet / "images"
            if not is_expanded
            else "expanded_images"
        )
    elif "r2f" in model.lower():
        images_dir = (
            Path(base_dir) / "ICCV_output" / "R2F" / "sdxl" / triplet / "images"
            if not is_expanded
            else "expanded_images"
        )
    elif "rrnet" in model.lower():
        image_folder_name = get_base_prompt_from_action_triplet(triplet)

        if "." in image_folder_name:
            assert False, f"folder Name shoulnd't haave '.' : {image_folder_name}"
        base_dir = "/ix/akovashka/sem238/RRNet"
        images_dir = (
            Path(base_dir)
            / "generation_result"
            / relation
            / image_folder_name
            / "w=0.6"
        )
    elif "sld" in model:
        images_dir = (
            Path(base_dir)
            / "outputs"
            / f"{model}_{triplet}_{triplet}"
            / "sld_baseline"
            / f"{triplet}_to_{triplet}"
            / model.split("_")[1]
            / "final_sld_images"
        )
    elif not is_pretrained:
        tmp = "use_freq_rare" if use_freq_rare else "intermediate"
        images_dir = (
            Path(base_dir)
            / "ICCV_output"
            / model
            / training_mode
            / rolebench_data[relation]["rare"]
            / tmp
            / "validation"
        )
        epoch_dir = get_latest_epoch_dir(images_dir)
        images_dir = (
            images_dir
            / epoch_dir
            / triplet
            / ("images" if not is_expanded else "expanded_images")
        )
    else:
        _, relation, _ = parse_action_triplet(triplet)
        images_dir = (
            Path(base_dir)
            / "data"
            / relation
            / triplet
            / model
            / ("images" if not is_expanded else "exapnded_images")
        )
    return images_dir


def get_all_triplets_for_relation_type(relation_type, relations):
    triplets = []
    for relation in relations:
        triplets.append(rolebench_data[relation][relation_type])
    return triplets


def get_metric_name(metric):
    if metric.lower() == "vqascore" or metric.lower() == "vqascore_ranking":
        return "VQAscore"
    elif metric.lower() == "qa_score":
        return "qa_score"
    else:
        raise ValueError(f"Metric {metric} not supported")


def summarize_resutls(
    metric,
    models,
    evaluator_names,
    categories,
    output_dir_base,
    is_pretrained,
    is_expanded_array=[],
    training_mode=None,
    use_freq_rare=[],
    top_k=None,
    relations=[],
):
    """Create CSV with averages across relation types"""
    print("\n=== Starting summarize_results ===")
    print(f"Models: {models}")
    print(f"Evaluator names: {evaluator_names}")
    print(f"Categories: {categories}")
    print(f"Top-k: {top_k}")

    # Create a metric name that includes top_k if applicable
    metric_with_k = metric
    if top_k is not None and metric.lower() == "vqascore_ranking":
        metric_with_k = f"{metric}_top{top_k}"

    base_dir = "/ix/akovashka/sem238/projects/SDAbstractSpatial"

    output_dir_base = output_dir_base / "results_summary" / metric_with_k
    output_dir_base = Path(output_dir_base)

    per_model_results = {}
    for model, is_expanded in zip(models, is_expanded_array):
        model_name = get_model_name(
            model, is_pretrained, is_expanded, training_mode, use_freq_rare
        )
        print(f"\nProcessing model: {model_name}")

        per_model_results[model_name] = {}
        for evaluator_name in evaluator_names:
            print(f"\n  Evaluator: {evaluator_name}")
            per_model_results[model_name][evaluator_name] = {}

            for relation in relations:
                print(f"\n    Relation: {relation}")
                for relation_type in ["rare", "frequent"]:
                    print(f"\n      Relation type: {relation_type}")
                    triplet = rolebench_data[relation][relation_type]
                    print(f"      Triplet: {triplet}")

                    # Initialize the dictionary ONCE for this triplet
                    per_model_results[model_name][evaluator_name][triplet] = {}
                    for col in categories + ["total"]:
                        per_model_results[model_name][evaluator_name][triplet][col] = {
                            "matching": None,
                            "unmatching": None,
                        }

                    eval_triplets = get_evaluation_triplets(triplet)
                    print(f"      Evaluation triplets: {eval_triplets}")

                    # For vqascore_ranking, we need to first get the top-k image numbers
                    top_k_image_nums = None
                    if metric.lower() == "vqascore_ranking" and top_k is not None:
                        # First, get the matching scores for the original triplet
                        for eval_triplet in eval_triplets:
                            if (
                                eval_triplet == triplet
                            ):  # Only process the matching triplet
                                validation_dir = get_model_validation_path(
                                    base_dir,
                                    model,
                                    evaluator_name,
                                    triplet,
                                    eval_triplet,
                                    metric.split("_")[
                                        0
                                    ],  # Use vqascore for the base metric
                                    is_expanded,
                                    is_pretrained,
                                    use_freq_rare,
                                    training_mode,
                                )

                                csv_path = (
                                    validation_dir
                                    / f"VQA_scores_{evaluator_name}_triplet_{eval_triplet}.csv"
                                )

                                if csv_path.exists():
                                    # Read the CSV and get the top-k image numbers based on total score
                                    df = pd.read_csv(csv_path)
                                    # Ensure 'image_num' column exists
                                    if "image_num" not in df.columns:
                                        print(
                                            f"WARNING: 'image_num' column not found in {csv_path}"
                                        )
                                        continue

                                    # Sort by total score and get top-k image numbers
                                    df = df.sort_values(by="total", ascending=False)
                                    top_k_image_nums = (
                                        df["image_num"].head(top_k).tolist()
                                    )
                                    print(
                                        f"        Selected top {top_k} images: {top_k_image_nums}"
                                    )
                                break

                    # Now process all eval_triplets
                    for eval_triplet in eval_triplets:
                        print(f"\n        Processing eval_triplet: {eval_triplet}")
                        print("MODEL IS ", model)
                        validation_dir = get_model_validation_path(
                            base_dir,
                            model,
                            evaluator_name,
                            triplet,
                            eval_triplet,
                            (
                                "vqascore"
                                if metric.lower() == "vqascore_ranking"
                                else metric
                            ),
                            is_expanded,
                            is_pretrained,
                            use_freq_rare,
                            training_mode,
                        )

                        if metric.lower() == "qa_score":
                            csv_path = (
                                validation_dir
                                / f"qa_scores_{evaluator_name}_triplet_{eval_triplet}.csv"
                            )
                        elif (
                            metric.lower() == "vqascore"
                            or metric.lower() == "vqascore_ranking"
                        ):
                            csv_path = (
                                validation_dir
                                / f"VQA_scores_{evaluator_name}_triplet_{eval_triplet}.csv"
                            )
                        else:
                            raise ValueError(f"Metric {metric} not supported")

                        print(f"        CSV path: {csv_path}")
                        if not csv_path.exists():
                            print(
                                f"        WARNING: CSV path does not exist: {csv_path}"
                            )
                            continue

                        # Read and process the CSV
                        df = pd.read_csv(csv_path)
                        print(f"        DataFrame shape: {df.shape}")

                        # For vqascore_ranking, filter to only include top-k images by image_num
                        if (
                            metric.lower() == "vqascore_ranking"
                            and top_k_image_nums is not None
                        ):
                            if "image_num" in df.columns:
                                # Filter to only include the top-k image numbers
                                df = df[df["image_num"].isin(top_k_image_nums)]
                                print(
                                    f"        Filtered to top-k images, new shape: {df.shape}"
                                )
                                if len(df) == 0:
                                    print(
                                        f"        WARNING: No matching images found after filtering"
                                    )
                                    continue
                            else:
                                print(
                                    f"        WARNING: 'image_num' column not found, cannot filter"
                                )

                        # Sort by total score
                        df = df.sort_values(by="total", ascending=False)

                        cols = [category for category in categories]
                        cols.append("total")

                        for col in cols:
                            print(f"\n          Processing column: {col}")
                            category_scores = df[col].tolist()
                            score = np.mean(category_scores)
                            print(f"          Calculated score: {score}")
                            print(f"          eval_triplet: {eval_triplet}")
                            print(f"          triplet: {triplet}")

                            if eval_triplet == triplet:
                                print(f"          Assigning matching score: {score}")
                                per_model_results[model_name][evaluator_name][triplet][
                                    col
                                ]["matching"] = score
                            elif eval_triplet == reverse_triplet(triplet):
                                print(f"          Assigning unmatching score: {score}")
                                per_model_results[model_name][evaluator_name][triplet][
                                    col
                                ]["unmatching"] = score
                            else:
                                print(f"          WARNING: Unknown triplet combination")
                                raise ValueError(
                                    f"Unknown triplet: {eval_triplet} for triplet {triplet}"
                                )

    print("\nFinal per_model_results structure:")
    print(json.dumps(per_model_results, indent=2))

    header1 = ["Model", f"Triplet Type"]
    header2 = ["", ""]

    triplet_types = ["frequent", "rare"]
    for evaluator_name in evaluator_names:
        cols = [category for category in categories]
        cols.append("total")
        for category in cols:
            header1.extend([f"{metric_with_k} ({evaluator_name})"] * 3)
            header2.extend(
                [
                    f"{category}_matching",
                    f"{category}_unmatching",
                    f"{category}_delta",
                ]
            )

    # Create rows
    rows = [header1, header2]

    for model, is_expanded in zip(models, is_expanded_array):
        model_name = get_model_name(
            model, is_pretrained, is_expanded, training_mode, use_freq_rare
        )
        output_dir = output_dir_base / model_name
        output_file = output_dir / f"{metric_with_k}_results.csv"
        os.makedirs(output_dir, exist_ok=True)

        for relation_type in triplet_types:
            row = [model_name, relation_type]
            all_triplets_per_relation_type = get_all_triplets_for_relation_type(
                relation_type, relations
            )
            for evaluator_name in evaluator_names:
                for category in categories + ["total"]:
                    matching = []
                    unmatching = []
                    delta = []
                    try:
                        for triplet in all_triplets_per_relation_type:
                            if per_model_results[model_name][evaluator_name][triplet][
                                category
                            ]:
                                matching.append(
                                    per_model_results[model_name][evaluator_name][
                                        triplet
                                    ][category]["matching"]
                                )
                                unmatching.append(
                                    per_model_results[model_name][evaluator_name][
                                        triplet
                                    ][category]["unmatching"]
                                )
                                delta.append(
                                    per_model_results[model_name][evaluator_name][
                                        triplet
                                    ][category]["matching"]
                                    - per_model_results[model_name][evaluator_name][
                                        triplet
                                    ][category]["unmatching"]
                                )
                        matching_avg = np.mean(matching)
                        unmatching_avg = np.mean(unmatching)
                        delta_avg = np.mean(delta)
                        row.extend(
                            [
                                f"{matching_avg:.3f}",
                                f"{unmatching_avg:.3f}",
                                f"{delta_avg:.3f}",
                            ]
                        )
                    except:
                        row.extend(["N/A", "N/A", "N/A"])

            rows.append(row)

        # Write to CSV
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        print(f"Results saved to {output_file}")




def summarize_resutls_per_category(
    metric,
    models,
    evaluator_names,
    categories,
    output_dir_base,
    is_pretrained,
    is_expanded_array=[],
    training_mode=None,
    use_freq_rare=[],
    top_k=None,
    relations=[],
):
    """Create CSV with averages across relation types"""
    print("\n=== Starting summarize_results PER CATEGORY ===")
    print(f"Models: {models}")
    print(f"Evaluator names: {evaluator_names}")
    print(f"Categories: {categories}")
    base_dir = "/ix/akovashka/sem238/projects/SDAbstractSpatial"
    base_dir = Path(base_dir)

    output_dir_base = output_dir_base / "results_summary" / metric
    output_dir_base = Path(output_dir_base)

    per_model_results = {}
    for model, is_expanded in zip(models, is_expanded_array):
        model_name = get_model_name(
            model, is_pretrained, is_expanded, training_mode, use_freq_rare
        )
        print(f"\nProcessing model: {model_name}")

        per_model_results[model_name] = {}
        for evaluator_name in evaluator_names:
            print(f"\n  Evaluator: {evaluator_name}")
            per_model_results[model_name][evaluator_name] = {}

            for relation in relations:
                print(f"\n    Relation: {relation}")
                for relation_type in ["rare", "frequent"]:
                    print(f"\n      Relation type: {relation_type}")
                    triplet = rolebench_data[relation][relation_type]
                    print(f"      Triplet: {triplet}")

                    # Initialize the dictionary ONCE for this triplet
                    per_model_results[model_name][evaluator_name][triplet] = {}
                    for col in categories + ["total"]:
                        per_model_results[model_name][evaluator_name][triplet][col] = {
                            "matching": None,
                            "unmatching": None,
                        }

                    eval_triplets = get_evaluation_triplets(triplet)
                    print(f"      Evaluation triplets: {eval_triplets}")

                    for eval_triplet in eval_triplets:
                        print(f"\n        Processing eval_triplet: {eval_triplet}")
                        print("MODEL IS ", model)
                        validation_dir = get_model_validation_path(
                            base_dir,
                            model,
                            evaluator_name,
                            triplet,
                            eval_triplet,
                            metric,
                            is_expanded,
                            is_pretrained,
                            use_freq_rare,
                            training_mode,
                        )

                        if metric == "qa_score":
                            csv_path = (
                                validation_dir
                                / f"qa_scores_{evaluator_name}_triplet_{eval_triplet}.csv"
                            )
                            print(f"        CSV path: {csv_path}")
                        elif metric.lower() == "vqascore":
                            csv_path = (
                                validation_dir
                                / f"VQA_scores_{evaluator_name}_triplet_{eval_triplet}.csv"
                            )
                            print(f"        CSV path: {csv_path}")
                        else:
                            raise ValueError(f"Metric {metric} not supported")

                        if not csv_path.exists():
                            print(
                                f"        WARNING: CSV path does not exist: {csv_path}"
                            )
                            continue
                            # raise ValueError(
                            #     f"Warning: Missing results for {eval_triplet}: {csv_path}"
                            # )

                        # Read and process the CSV
                        df = pd.read_csv(csv_path)
                        print(f"        DataFrame shape: {df.shape}")

                        df = df.sort_values(by="total", ascending=False)
                        cols = [category for category in categories]
                        cols.append("total")

                        for col in cols:
                            print(f"\n          Processing column: {col}")
                            # if col == "total":
                            # if triplet == "mouse_chasing_cat":
                            #     print(
                            #         "score is  category_scores  ", category_scores
                            #     )
                            category_scores = df[col].tolist()
                            score = (
                                np.mean(category_scores)
                                if top_k is None
                                else np.mean(category_scores[:top_k])
                            )

                            print(f"          Calculated score: {score}")
                            print(f"          eval_triplet: {eval_triplet}")
                            print(f"          triplet: {triplet}")

                            if eval_triplet == triplet:
                                print(f"          Assigning matching score: {score}")
                                per_model_results[model_name][evaluator_name][triplet][
                                    col
                                ]["matching"] = score
                            elif eval_triplet == reverse_triplet(triplet):
                                print(f"          Assigning unmatching score: {score}")
                                per_model_results[model_name][evaluator_name][triplet][
                                    col
                                ]["unmatching"] = score
                            else:
                                print(f"          WARNING: Unknown triplet combination")
                                raise ValueError(
                                    f"Unknown triplet: {eval_triplet} for triplet {triplet}"
                                )

    print("\nFinal per_model_results structure:")
    print(json.dumps(per_model_results, indent=2))

    header1 = ["Model", f"tripet"]
    header2 = ["", ""]

    for evaluator_name in evaluator_names:

        cols = [category for category in categories]
        cols.append("total")
        for category in cols:
            header1.extend([f"{metric} ({evaluator_name})"] * 3)
            header2.extend(
                [
                    f"{category}_matching",
                    f"{category}_unmatching",
                    f"{category}_delta",
                ]
            )

    # Create rows
    rows = [header1, header2]

    for model, is_expanded in zip(models, is_expanded_array):
        model_name = get_model_name(
            model, is_pretrained, is_expanded, training_mode, use_freq_rare
        )
        output_dir = output_dir_base / model_name
        output_file = output_dir / f"{metric}_per_triplet_results.csv"
        os.makedirs(output_dir, exist_ok=True)

        for relation in relations:
            for relation_type in ["rare", "frequent"]:
                triplet = rolebench_data[relation][relation_type]
                row = [model_name, triplet]
                for evaluator_name in evaluator_names:

                    for category in categories + ["total"]:
                        matching = []
                        unmatching = []
                        delta = []
                        try:
                            if per_model_results[model_name][evaluator_name][triplet][
                                category
                            ]:
                                matching.append(
                                    per_model_results[model_name][evaluator_name][
                                        triplet
                                    ][category]["matching"]
                                )
                                unmatching.append(
                                    per_model_results[model_name][evaluator_name][
                                        triplet
                                    ][category]["unmatching"]
                                )
                                delta.append(
                                    per_model_results[model_name][evaluator_name][
                                        triplet
                                    ][category]["matching"]
                                    - per_model_results[model_name][evaluator_name][
                                        triplet
                                    ][category]["unmatching"]
                                )
                            matching_avg = np.mean(matching)
                            unmatching_avg = np.mean(unmatching)
                            delta_avg = np.mean(delta)
                            row.extend(
                                [
                                    f"{matching_avg:.3f}",
                                    f"{unmatching_avg:.3f}",
                                    f"{delta_avg:.3f}",
                                ]
                            )
                        except:
                            row.extend(["N/A", "N/A", "N/A"])

                rows.append(row)

        # Write to CSV
        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        print(f"Results saved to {output_file}")


# Usage example:
# create_comparison_csv(
#     model_names=['dall-e-3', 'sdxl'],
#     evaluator_names=['llava', 'internvl'],
#     categories=['detection', 'spatial', 'pose', 'orientation', 'interaction', 'count']
# )

if __name__ == "__main__":
    # # Configure basic logging
    # logging.basicConfig(level=logging.INFO, format="%(message)s")

    # # Get command line arguments
    # if len(sys.argv) != 4:
    #     print("Usage: python eval.py <model_name> <evaluator_name> <output_file>")
    #     sys.exit(1)

    # model_name = sys.argv[1]
    # evaluator_name = sys.argv[2]
    # output_file = sys.argv[3]

    # metrics = analyze_all_results(
    #     model_name=model_name, evaluator_name=evaluator_name, output_file=output_file
    # )
    # &*++***************************
    # create_comparison_csv(
    #     model_names=["dall-e-3", "VQAScore_clip-flant5-xxl"],
    #     evaluator_names=["VQAScore_gpt-4o"],
    #     categories=[
    #         "detection",
    #         "spatial",
    #         "pose",
    #         "orientation",
    #         "interaction",
    #         "count",
    #     ],
    # )

    # create_relation_type_averages_csv(
    #     model_names=["dall-e-3", "sdxl", "sd3-5", "auraflow2"],
    #     evaluator_names=["VQAScore_clip-flant5-xxl"],
    #     categories=[
    #         "detection",
    #         "spatial",
    #         "pose",
    #         "orientation",
    #         "interaction",
    #         "count",
    #     ],
    # )

    # "VQAScore_gpt-4o"
    # internvl", "VQAScore_clip-flant5-xxl"

    # create_relation_type_averages_csv(
    #     model_names=[
    #         "sup_only",  # matches the directory structure: ICCV_output/sdxl_lora/sup_only/...
    #     ],
    #     evaluator_names=["VQAScore_clip-flant5-xxl"],
    #     categories=[
    #         "detection",
    #         "spatial",
    #         "pose",
    #         "orientation",
    #         "interaction",
    #         "count",
    #     ],
    #     output_file="avg_relations.csv",
    #     use_freq_rare=False,
    # )
    #  [
    #             "detection",
    #             "count",
    #             "spatial",
    #             "pose",
    #             "orientation",
    #             "interaction",
    #         ]
    # summarize_resutls(
    #     "qa_score",
    #     ["dall-e-3", "sdxl", "sd3-5", "auraflow2", "itercomp"],
    #     ["VQAScore_clip-flant5-xxl"],
    #     ["detection", "count", "spatial", "orientation", "pose", "interaction"],
    #     Path("/ix/akovashka/sem238/projects/SDAbstractSpatial/ICCV_output/"),
    #     is_pretrained=True,
    #     is_expanded_array=[True] * 5,
    #     training_mode="sup_only",
    #     use_freq_rare=False,
    #     top_k=None,
    # )

    print("JISSSSS")
    RELATIONS_TO_CHECK = [
        "chasing",
        "riding",
        "throwing",
        "holding",
        "following",
        "feeding",
        "pulling",
        "lifting",
        "carrying",
        "kissing",
    ]
    metric = "VQAscore"  # VQAscore or qa_score or VQAscore_ranking
    if metric.lower() == "vqascore" or metric.lower() == "vqascore_ranking":
        categories = []
    else:
        categories = [
            "detection",
            "count",
            "spatial",
            "orientation",
            "pose",
            "interaction",
        ]
    # models = ["dall-e-3", "sdxl", "sd3", "sd3-5", "auraflow2", "itercomp"]
    models = ["sld_sdxl"]
    # models = ["RRNET", "R2F", "sld_dall-e-3", "sld_sdxl", "sdxl_lora"]
    is_pretrained = False
    is_expanded = False
    use_freq_rare = False
    top_k = -1
    training_mode = "sup_only_weighted_sup_ActiveWeight5.0"

    summarize_resutls(
        metric,
        models,
        ["VQAScore_clip-flant5-xxl"],
        categories,
        Path("/ix/akovashka/sem238/projects/SDAbstractSpatial/ICCV_rebuttal/"),
        is_pretrained=is_pretrained,
        is_expanded_array=[is_expanded] * len(models),
        training_mode=training_mode,
        use_freq_rare=use_freq_rare,
        top_k=top_k,
        relations=RELATIONS_TO_CHECK,
    )

    # summarize_resutls_per_category(
    #     metric,
    #     models,
    #     ["VQAScore_clip-flant5-xxl"],
    #     categories,
    #     Path("/ix/akovashka/sem238/projects/SDAbstractSpatial/ICCV_output/"),
    #     is_pretrained=is_pretrained,
    #     is_expanded_array=[is_expanded] * len(models),
    #     training_mode=training_mode,
    #     use_freq_rare=use_freq_rare,
    #     top_k=top_k,
    #     relations=RELATIONS_TO_CHECK,
    # )


