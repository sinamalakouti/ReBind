import json
import logging
import os
from datetime import datetime
from pathlib import Path
import pandas as pd

import numpy as np
import csv
from collections import defaultdict

from dataset.rolebench import (
    get_evaluation_triplets,
    parse_action_triplet,
    reverse_triplet,
    rolebench_data,
    RELATIONS
)
from dataset.paths import get_data_dir
from prompts.prompt_utils import (
    get_base_prompt_from_action_triplet,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



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
    if "sdxl_lora" in str(output_dir) or "stable-diffusion-3-medium" in str(output_dir):
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
                    print("metric_dir does not exist: ", metric_dir)
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
    print(f"GET EPOCH DIR,  BASE_PATH: {base_path}")
    epoch_dirs = [d for d in base_path.glob("epoch_*") if d.is_dir()]
    if not epoch_dirs:
        logger.warning(f"WARNING: No epoch directories found in {base_path}")
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
    input_dir_base_name,
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
            / input_dir_base_name  # ICCV_output
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
            / input_dir_base_name  # ICCV_output
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
            / input_dir_base_name  # ICCV_output
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
            / input_dir_base_name  # outputs
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
            / input_dir_base_name  # ICCV_output
            / model
            / training_mode  ##(training_mode +rolebench_data[relation]["rare"])
            / rolebench_data[relation]["rare"]
            / tmp
            / "validation"
        )
        validation_dir = Path(validation_dir)
        epoch_dir = get_latest_epoch_dir(validation_dir)
        print(
            f"VALIDATION DIR: {validation_dir}, EPOCH DIR: {epoch_dir}, TRIPLET: {triplet}, IS EXPANDED: {is_expanded}, IS PRETRAINED: {is_pretrained}, USE FREQ RARE: {use_freq_rare}, TRAINING MODE: {training_mode}"
        )
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


def summarize_results(
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
    input_dir_base_name="ICCV_output",
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
        # training_mode_orig = training_mode
        per_model_results[model_name] = {}
        for evaluator_name in evaluator_names:
            print(f"\n  Evaluator: {evaluator_name}")
            per_model_results[model_name][evaluator_name] = {}

            for relation in relations:
                # training_mode = training_mode_orig  + rolebench_data[relation]['rare']
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
                                    input_dir_base_name,
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
                            input_dir_base_name,
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
                    except Exception as e:
                        # error_msg = str(e)
                        # raise ValueError(f"Error processing {model_name}/{evaluator_name}/{category}: {error_msg}")
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
                            input_dir_base_name,
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
    print("YAAAY")
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
    metric = "qa_score"  # VQAscore or qa_score or VQAscore_ranking
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
    models = ["stable-diffusion-3-medium"]
    # models = ["RRNET", "R2F", "sld_dall-e-3", "sld_sdxl", "sdxl_lora"]
    is_pretrained = False  # set it to True if you want to use pre-trained models such as dall-e-3, sdxl, etc.
    is_expanded = False  # i think it only make sense for when is_pretrained is True
    use_freq_rare = False  # set it to True if you want to use freq/rare relations, this only make sense if runnign sdxl_lora
    top_k = -1  # this is relevant only for VQAscore_ranking
    list_num_example_per_intermediate = ["full"]
    for num_example_per_intermediate in list_num_example_per_intermediate:
        # training_mode = f"sup_only_weighted_sup_ActiveWeight5.0_numExamplePerIntermediate{num_example_per_intermediate}_numIntermediatesPerRelationfull_intermediateTypes" #"sup_only_weighted_sup_ActiveWeight5.0"
        training_mode = f"sup_only_weighted_sup_ActiveWeight5.0_numExamplePerIntermediate{num_example_per_intermediate}_numIntermediatesPerRelationfull_intermediateTypesactive_passive"  # "sup_only_weighted_sup_ActiveWeight5.0"

        # CONFERENCE = ICCV_output, ICCV_rebuttal
        summarize_results(
            metric,
            models,
            ["VQAScore_clip-flant5-xxl"],
            categories,
            Path("<YOUR-BASE-PATH>/<FOLDER-NAME>/"),
            is_pretrained=is_pretrained,
            is_expanded_array=[is_expanded] * len(models),
            training_mode=training_mode,
            use_freq_rare=use_freq_rare,
            top_k=top_k,
            relations=RELATIONS_TO_CHECK,
            input_dir_base_name="<FOLDER-NAME>",
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


