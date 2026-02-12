from typing import List
from math import log
from re import sub
import pandas as pd
import os

from sympy import use


def parse_metrics_and_print_averages(
    log_dir,
    log_dir_qa_score,
    triplet,
    MLLM_evaluator_metric,
    metrics,
    use_VQA_score=True,
):
    """Parse metrics CSV files and handle both tensor and regular formats with error handling"""
    combined_df = None

    # Print log directories for each metric
    print("\nLog directories:")
    print("-" * 30)
    for metric in metrics:
        current_log_dir = log_dir_qa_score if metric == "qa_score" else log_dir
        print(f"{metric:25}: {current_log_dir}")
    print("-" * 30)

    for metric in metrics:
        # Use qa_score_root_log_dir for qa_score metric
        current_log_dir = log_dir_qa_score if metric == "qa_score" else log_dir

        file_path = os.path.join(
            current_log_dir,
            f"mllm_{MLLM_evaluator_metric[metric]}",
            f"{metric}_MLLM_evaluator_{MLLM_evaluator_metric[metric]}_metrics_{triplet}.csv",
        )

        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            print("df", df)
            if combined_df is None:
                combined_df = df[["image_num"]]

            # Process each column
            metric_columns = [col for col in df.columns if col != "image_num"]
            for col in metric_columns:
                if use_VQA_score or df[col].dtype == object:

                    def extract_value(x):
                        try:
                            # Handle tensor format
                            if "tensor" in str(x):
                                return float(str(x).split("(")[1].split(",")[0])
                            # Handle regular number in string format
                            return float(x)
                        except Exception as e:
                            print(f"Error processing value: {x}")
                            print(f"Error details: {e}")
                            return float("nan")  # Return NaN for problematic values

                    values = df[col].apply(extract_value)
                else:
                    values = df[col]

                # Add to combined DataFrame
                col_name = f"{metric}_{col}" if col != metric else metric
                combined_df[col_name] = values
        except Exception as e:
            print(f"Warning: Could not process {metric} for {triplet}. Error: {e}")
            continue

    # Calculate and print averages for each metric
    print(f"\nResults for {triplet}:")
    print("-" * 30)

    if "qa_score" in metrics and "qa_score" in combined_df.columns:
        print(f"{'qa_score':25}: {combined_df['qa_score'].mean():.4f}")

    if "rel_score" in metrics and "rel_score" in combined_df.columns:
        print(f"{'rel_score':25}: {combined_df['rel_score'].mean():.4f}")

    if "entropy_score" in metrics:
        if "entropy_score_total_entropy" in combined_df.columns:
            print(
                f"{'entropy_score_total':25}: {combined_df['entropy_score_total_entropy'].mean():.4f}"
            )
        if "entropy_score_true_relation_entropy" in combined_df.columns:
            print(
                f"{'entropy_score_true_rel':25}: {combined_df['entropy_score_true_relation_entropy'].mean():.4f}"
            )

    return combined_df


def save_seed_averages(
    log_dir: str,
    qa_score_root_log_dir: str,
    triplet: str,
    seed_results: dict,
    MLLM_evaluator_metric: dict,
    metrics: List[str],
):
    """Save averaged results across seeds"""
    # Convert all seed results to DataFrames and concatenate
    all_seed_dfs = []
    for seed, df in seed_results.items():
        all_seed_dfs.append(df)

    # Calculate mean across all seeds
    avg_df = pd.concat(all_seed_dfs).groupby("image_num").mean().reset_index()

    # Save each metric separately under its corresponding evaluator
    for metric in metrics:
        if metric == "qa_score" and "qa_score" in avg_df.columns:
            avg_file_dir = os.path.join(
                qa_score_root_log_dir,
                f"mllm_{MLLM_evaluator_metric['qa_score']}",
            )

            os.makedirs(avg_file_dir, exist_ok=True)

            avg_file_path = os.path.join(
                avg_file_dir,
                f"avg_qa_score_MLLM_evaluator_{MLLM_evaluator_metric['qa_score']}_metrics_{triplet}.csv",
            )
            qa_df = avg_df[["image_num", "qa_score"]]
            qa_df.to_csv(avg_file_path, index=False)

        elif metric == "rel_score" and "rel_score" in avg_df.columns:
            avg_file_dir = os.path.join(
                log_dir,
                f"mllm_{MLLM_evaluator_metric['qa_score']}",
            )

            os.makedirs(avg_file_dir, exist_ok=True)

            rel_score_path = os.path.join(
                avg_file_dir,
                f"avg_rel_score_MLLM_evaluator_{MLLM_evaluator_metric['rel_score']}_metrics_{triplet}.csv",
            )
            rel_score_df = avg_df[["image_num", "rel_score"]]
            rel_score_df.to_csv(rel_score_path, index=False)

        elif (
            metric == "entropy_score"
            and "entropy_score_total_entropy" in avg_df.columns
        ):
            avg_file_dir = os.path.join(
                log_dir,
                f"mllm_{MLLM_evaluator_metric['qa_score']}",
            )

            os.makedirs(avg_file_dir, exist_ok=True)

            entropy_score_path = os.path.join(
                avg_file_dir,
                f"avg_entropy_score_MLLM_evaluator_{MLLM_evaluator_metric['entropy_score']}_metrics_{triplet}.csv",
            )
            entropy_score_df = avg_df[
                [
                    "image_num",
                    "entropy_score_total_entropy",
                    "entropy_score_true_relation_entropy",
                ]
            ]
            entropy_score_df.to_csv(entropy_score_path, index=False)


def generate_final_predictions(
    root_log_dir: str,
    action_rel_triplets: List[str],
    seeds: List[int],
    MLLM_evaluator_metric: dict,
    metrics: List[str],
):
    """Generate final predictions using averaged scores"""
    predictions_data = []
    print("FINAL PATH LOG IS ", root_log_dir)

    if "qa_score" in metrics:
        print("\nGenerating predictions...")
        print("-" * 50)

        for i in range(0, len(action_rel_triplets), 2):
            freq_prompt = action_rel_triplets[i]
            rare_prompt = action_rel_triplets[i + 1]

            freq_scores_path = os.path.join(
                root_log_dir,
                f"mllm_{MLLM_evaluator_metric['qa_score']}",
                f"avg_qa_score_MLLM_evaluator_{MLLM_evaluator_metric['qa_score']}_metrics_{freq_prompt}.csv",
            )
            rare_scores_path = os.path.join(
                root_log_dir,
                f"mllm_{MLLM_evaluator_metric['qa_score']}",
                f"avg_qa_score_MLLM_evaluator_{MLLM_evaluator_metric['qa_score']}_metrics_{rare_prompt}.csv",
            )

            if os.path.exists(freq_scores_path) and os.path.exists(rare_scores_path):
                freq_scores = pd.read_csv(freq_scores_path)
                rare_scores = pd.read_csv(rare_scores_path)

                for idx in range(len(freq_scores)):
                    delta = (
                        freq_scores.iloc[idx]["qa_score"]
                        - rare_scores.iloc[idx]["qa_score"]
                    )
                    pred = freq_prompt if delta >= 0 else rare_prompt

                    predictions_data.append(
                        {
                            "image_num": freq_scores.iloc[idx]["image_num"],
                            "freq_prompt": freq_prompt,
                            "rare_prompt": rare_prompt,
                            "delta": delta,
                            "pred": pred,
                        }
                    )

        if predictions_data:
            predictions_df = pd.DataFrame(predictions_data)
            predictions_path = os.path.join(root_log_dir, "final_predictions.csv")
            predictions_df.to_csv(predictions_path, index=False)
            print(f"\nSaved final predictions to: {predictions_path}")


# Example usage:
if __name__ == "__main__":
    # Example directory and triplets
    # subject = "mouse"
    # object = "cat"
    # gen_prompt = f"A photo of a {subject} chasing a {object}"
    # guidance_prompt = "mouse_chasing_cat"
    # action_rel_triplets = ["cat_chasing_mouse", "mouse_chasing_cat"]

    # ________________________________________________________________________________________

    # object = "an astronaut"
    # subject = "a horse"
    # gen_prompt = f"A photo of {subject} riding on {object}"
    # # guidance_prompt = "astronaut_riding_horse"
    # guidance_prompt = "horse_riding_astronaut"
    # action_rel_triplets = ["astronaut_riding_horse", "horse_riding_astronaut"]
    # ________________________________________________________________________________________

    object = "boy"
    subject = "puppy"
    gen_prompt = f"A photo of a {subject} throwing a ball for a {object}"
    guidance_prompt = "boy_throwing_puppy"
    guidance_prompt = "puppy_throwing_boy"

    action_rel_triplets = ["boy_throwing_puppy", "puppy_throwing_boy"]
    # ________________________________________________________________________________________

    subject = "man"
    object = "horse"
    # gen_prompt = f"A photo of a {subject} feeding a {object}"
    gen_prompt = f"A photo of a {object} feeding food to a {subject}"
    # guidance_prompt = "man_feeding_horse"
    guidance_prompt = "horse_feeding_man"

    action_rel_triplets = ["man_feeding_horse", "horse_feeding_man"]
    # ________________________________________________________________________________________

    # gen_prompt = f"A photo of a {subject} chasing a {object}"
    # gen_prompt = f"A photo of a {subject} chasing a {object}"
    # gen_prompt = f"A photo of a {subject} chasing a {object}"
    # gen_prompt = f"A photo of a {subject} chasing a {object}"

    n_ctx = 8
    root_log_dir = f"output/diffusion-dpo-softPrompt/INITPROMPT{gen_prompt.replace(' ', '_')}_NCTX{n_ctx}_EVALGOAL{guidance_prompt.replace(' ', '_')}_MLLMinternvl_REWARDqa_score/final_results"
    # "dall-e-3"  # "IterComp"  # "RPG_sdxl" #stable-diffusion-xl-base-1.0
    # model = "RPG_sdxl"
    # model = "IterComp"
    # model = "stable-diffusion-xl-base-1.0"
    print("Generation prompt: ", gen_prompt)
    print("Guidance prompt: ", guidance_prompt)
    print("n_ctx: ", n_ctx)

    metrics = ["qa_score"]
    use_VQA_scores = False
    triplet = "puppy_throwing_boy"
    relation = triplet.split("_")[1]
    result_dir_name = "results_expanded_prompt"
    # result_dir_name = "results"
    model = "sd3-5"
    action_rel_triplets_dict = {
        "cat_chasing_mouse": ["cat_chasing_mouse", "mouse_chasing_cat"],
        "mouse_chasing_cat": ["cat_chasing_mouse", "mouse_chasing_cat"],
        "astronaut_riding_horse": ["astronaut_riding_horse", "horse_riding_astronaut"],
        "horse_riding_astronaut": ["astronaut_riding_horse", "horse_riding_astronaut"],
        "boy_throwing_puppy": ["boy_throwing_puppy", "puppy_throwing_boy"],
        "puppy_throwing_boy": ["boy_throwing_puppy", "puppy_throwing_boy"],
        "man_feeding_horse": ["man_feeding_horse", "horse_feeding_man"],
        "horse_feeding_man": ["man_feeding_horse", "horse_feeding_man"],
        "woman_feeding_baby": ["women_feeding_baby", "baby_feeding_women"],
        "baby_feeding_woman": ["woman_feeding_baby", "baby_feeding_woman"],
    }
    root_log_dir = f"/ix/akovashka/sem238/projects/SDAbstractSpatial/data/{model}/{relation}/{triplet}/{result_dir_name}"
    # root_log_dir = f"/ix/akovashka/sem238/image_score_pairs/init_prompts/{model}/A_photo_of_a_mouse_chasing_a_cat"
    action_rel_triplets = action_rel_triplets_dict[triplet]

    # Data path and triplets (init_prompts)

    # root_log_dir = f"/ix/akovashka/sem238/image_score_pairs/init_prompts/{model}/A_photo_of_a_cat_chasing_a_mouse"
    # # root_log_dir = f"/ix/akovashka/sem238/image_score_pairs/init_prompts/{model}/A_photo_of_a_mouse_chasing_a_cat"
    # action_rel_triplets = ["cat_chasing_mouse", "mouse_chasing_cat"]
    #
    # root_log_dir = f"/ix/akovashka/sem238/image_score_pairs/init_prompts/{model}/A_photo_of_an_astronaut_riding_on_a_horse"
    # root_log_dir = f"/ix/akovashka/sem238/image_score_pairs/init_prompts/{model}/A_photo_of_a_horse_riding_on_an_astronaut"
    # action_rel_triplets = ["astronaut_riding_horse", "horse_riding_astronaut"]

    # root_log_dir = f"/ix/akovashka/sem238/image_score_pairs/init_prompts/{model}/A_photo_of_a_boy_throwing_a_ball_for_a_puppy"
    # # root_log_dir = f"/ix/akovashka/sem238/image_score_pairs/init_prompts/{model}/A_photo_of_a_puppy_throwing_a_ball_for_a_boy"
    # action_rel_triplets = ["boy_throwing_puppy", "puppy_throwing_boy"]

    # root_log_dir = f"/ix/akovashka/sem238/image_score_pairs/init_prompts/{model}/A_photo_of_a_man_feeding_a_horse"
    # root_log_dir = f"/ix/akovashka/sem238/image_score_pairs/init_prompts/{model}/A_photo_of_a_horse_feeding_food_to_a_man"
    # action_rel_triplets = ["man_feeding_horse", "horse_feeding_man"]

    ################### Data path and triplets (SLD)

    # root_log_dir = f"outputs/sld_dall-e-3_mouse_chasing_cat_mouse_chasing_cat/mouse_chasing_cat_to_mouse_chasing_cat/"
    # action_rel_triplets = ["cat_chasing_mouse", "mouse_chasing_cat"]
    #
    # root_log_dir = f"outputs/sld_dall-e-3_horse_riding_astronaut_horse_riding_astronaut/horse_riding_astronaut_to_horse_riding_astronaut/"
    # action_rel_triplets = ["astronaut_riding_horse", "horse_riding_astronaut"]

    # root_log_dir = f"outputs/sld_dall-e-3_puppy_throwing_boy_puppy_throwing_boy/puppy_throwing_boy_to_puppy_throwing_boy/"
    # action_rel_triplets = ["boy_throwing_puppy", "puppy_throwing_boy"]

    # root_log_dir = f"outputs/sld_dall-e-3_horse_feeding_man_horse_feeding_man/horse_feeding_man_to_horse_feeding_man/"
    # action_rel_triplets = ["man_feeding_horse", "horse_feeding_man"]

    ################### Data path and triplets (RRNeT)
    # root_log_dir = f"/ix/akovashka/sem238/RRNet/generation_result/chasing/there_is_a_dog_chasing_a_cat/w06/"
    # root_log_dir = f"/ix/akovashka/sem238/RRNet/generation_result/chasing/there_is_a_cat_chasing_a_dog/w06/"
    # action_rel_triplets = ["dog_chasing_cat", "cat_chasing_dog"]
    # #
    # root_log_dir = f"outputs/sld_dall-e-3_horse_riding_astronaut_horse_riding_astronaut/horse_riding_astronaut_to_horse_riding_astronaut/"
    # action_rel_triplets = ["astronaut_riding_horse", "horse_riding_astronaut"]

    # root_log_dir = f"outputs/sld_dall-e-3_puppy_throwing_boy_puppy_throwing_boy/puppy_throwing_boy_to_puppy_throwing_boy/"
    # action_rel_triplets = ["boy_throwing_puppy", "puppy_throwing_boy"]

    # root_log_dir = f"outputs/sld_dall-e-3_horse_feeding_man_horse_feeding_man/horse_feeding_man_to_horse_feeding_man/"
    # action_rel_triplets = ["man_feeding_horse", "horse_feeding_man"]

    if use_VQA_scores:
        qa_score_root_log_dir = root_log_dir
        root_log_dir = os.path.join(root_log_dir, "VQAScore")
        qa_score_root_log_dir = root_log_dir
    else:
        qa_score_root_log_dir = root_log_dir
        root_log_dir = os.path.join(root_log_dir, "VQAScore")
        print("1111 qa_score_root_log_dir11111 : ", qa_score_root_log_dir)

    # MLLM_evaluator = "internvl"
    if use_VQA_scores:
        MLLM_evaluator_metric = {
            "qa_score": "clip-flant5-xxl",
            "rel_score": "llava",
            "entropy_score": "llava",
        }
    else:
        MLLM_evaluator_metric = {
            "qa_score": "internvl",
            "rel_score": "llava",
            "entropy_score": "llava",
        }
    print("\nLog directories for each metric:")
    print("-" * 50)
    for metric in metrics:
        current_log_dir = (
            qa_score_root_log_dir if metric == "qa_score" else root_log_dir
        )
        print(f"{metric:25}: {current_log_dir}")

    print("\nAnalyzing metrics for each triplet:")
    print("-" * 50)
    results = {}
    seeds = [1]
    seeds = [None]
    for triplet in action_rel_triplets:
        print(f"\nResults for {triplet}:")
        print("=" * 50)

        seed_results = {}
        all_metrics_results = {}

        for seed in seeds:
            print(f"\nSeed {seed}:")
            print("-" * 30)

            if seed is not None:
                log_dir = os.path.join(root_log_dir, f"seed_{seed}")
                log_dir_qa_score = os.path.join(qa_score_root_log_dir, f"seed_{seed}")
                print("log_dir_qa_score ", log_dir_qa_score)
            else:
                log_dir = root_log_dir
                log_dir_qa_score = qa_score_root_log_dir

            combined_results = parse_metrics_and_print_averages(
                log_dir,
                log_dir_qa_score,
                triplet,
                MLLM_evaluator_metric,
                metrics,
                use_VQA_scores,
            )

            # Store full results for averaging
            seed_results[seed] = combined_results

        # Save averaged results
        save_seed_averages(
            root_log_dir,
            qa_score_root_log_dir,
            triplet,
            seed_results,
            MLLM_evaluator_metric,
            metrics,
        )

    # Modify generate_final_predictions to use averaged files

    generate_final_predictions(
        qa_score_root_log_dir,
        action_rel_triplets,
        seeds,
        MLLM_evaluator_metric,
        metrics,
    )
