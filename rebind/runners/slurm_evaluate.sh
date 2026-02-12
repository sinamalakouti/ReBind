#!/bin/bash
#SBATCH --job-name=parallel_jobs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s
#SBATCH --time=0-00:10:00
#SBATCH --output=eval_slurm_logs/slurm-%A_%a.out
#SBATCH --error=eval_slurm_logs/slurm-%A_%a.err
#SBATCH --array=0-9

# Initialize empty PARAMS array
PARAMS=()
METRICS="'qa_score'" 
# Define parameter sets
# Baselines
# MODELS=("stable-diffusion-xl-base-1.0" "IterComp" "RPG_sdxl")
MODELS=("auraflow2")
USE_EXPANDED=true  # or false

for MODEL in "${MODELS[@]}"; do
    echo ${MODEL}
    # Convert boolean to flag or empty string
    expanded_flag=$([ "$USE_EXPANDED" = true ] && echo "--expanded_images" || echo "")
    
    PARAMS+=(
        "data/${MODEL}/chasing/mouse_chasing_cat/   'cat_chasing_mouse mouse_chasing_cat' internvl ${METRICS} ${expanded_flag}"
        "data/${MODEL}/chasing/cat_chasing_mouse/   'cat_chasing_mouse mouse_chasing_cat' internvl ${METRICS} ${expanded_flag}"
        "data/${MODEL}/riding/horse_riding_astronaut/   'astronaut_riding_horse horse_riding_astronaut' internvl ${METRICS} ${expanded_flag}"
        "data/${MODEL}/riding/astronaut_riding_horse/   'astronaut_riding_horse horse_riding_astronaut' internvl ${METRICS} ${expanded_flag}"
        "data/${MODEL}/throwing/boy_throwing_puppy/   'boy_throwing_puppy puppy_throwing_boy' internvl ${METRICS} ${expanded_flag}"
        "data/${MODEL}/throwing/puppy_throwing_boy/   'boy_throwing_puppy puppy_throwing_boy' internvl ${METRICS} ${expanded_flag}"
        "data/${MODEL}/feeding/man_feeding_horse/   'man_feeding_horse horse_feeding_man' internvl ${METRICS} ${expanded_flag}"
        "data/${MODEL}/feeding/horse_feeding_man/   'man_feeding_horse horse_feeding_man' internvl ${METRICS} ${expanded_flag}"  
        "data/${MODEL}/feeding/baby_feeding_women/   'baby_feeding_women women_feeding_baby' internvl ${METRICS} ${expanded_flag}"
        "data/${MODEL}/feeding/women_feeding_baby/   'baby_feeding_women women_feeding_baby' internvl ${METRICS} ${expanded_flag}"  
    )
done


# for MODEL in "${MODELS[@]}"; do
#     echo ${MODEL}
#     PARAMS+=(
#         "../../image_score_pairs/init_prompts/${MODEL}/A_photo_of_a_man_feeding_a_horse/   'man_feeding_horse horse_feeding_man' internvl ${METRICS}"
#         "../../image_score_pairs/init_prompts/${MODEL}/A_photo_of_a_horse_feeding_food_to_a_man/   'man_feeding_horse horse_feeding_man' internvl ${METRICS}"
#         # "../../image_score_pairs/init_prompts/${MODEL}/A_photo_of_a_mouse_chasing_a_cat/   'cat_chasing_mouse mouse_chasing_cat' internvl ${METRICS}"
#         # "../../image_score_pairs/init_prompts/${MODEL}/A_photo_of_a_cat_chasing_a_mouse/   'cat_chasing_mouse mouse_chasing_cat' internvl ${METRICS}"
#         # "../../image_score_pairs/init_prompts/${MODEL}/A_photo_of_a_boy_throwing_a_ball_for_a_puppy/    'boy_throwing_puppy puppy_throwing_boy' internvl ${METRICS}"
#         # "../../image_score_pairs/init_prompts/${MODEL}/A_photo_of_a_puppy_throwing_a_ball_for_a_boy/   'boy_throwing_puppy puppy_throwing_boy' internvl ${METRICS}"
#         # "../../image_score_pairs/init_prompts/${MODEL}/A_photo_of_an_astronaut_riding_on_a_horse/   'astronaut_riding_horse horse_riding_astronaut' internvl ${METRICS}"
#         # "../../image_score_pairs/init_prompts/${MODEL}/A_photo_of_a_horse_riding_on_an_astronaut/   'astronaut_riding_horse horse_riding_astronaut' internvl ${METRICS}"
#     )
# done


#soft prompt params: 


# PARAMS+=(
#         "output/diffusion-dpo-softPrompt/INITPROMPTA_photo_of_a_man_feeding_a_horse_NCTX8_EVALGOALman_feeding_horse_MLLMinternvl_REWARDqa_score/final_results/   'man_feeding_horse horse_feeding_man' internvl ${METRICS}"
#         "output/diffusion-dpo-softPrompt/INITPROMPTA_photo_of_a_horse_feeding_food_to_a_man_NCTX8_EVALGOALhorse_feeding_man_MLLMinternvl_REWARDqa_score/final_results/   'man_feeding_horse horse_feeding_man' internvl ${METRICS}"
#         "output/diffusion-dpo-softPrompt/INITPROMPTA_photo_of_a_cat_chasing_a_mouse_NCTX8_EVALGOALcat_chasing_mouse_MLLMinternvl_REWARDqa_score/final_results/  'cat_chasing_mouse mouse_chasing_cat' internvl ${METRICS}"
#         "output/diffusion-dpo-softPrompt/INITPROMPTA_photo_of_a_mouse_chasing_a_cat_NCTX8_EVALGOALmouse_chasing_cat_MLLMinternvl_REWARDqa_score/final_results/   'cat_chasing_mouse mouse_chasing_cat' internvl ${METRICS}"
#         "output/diffusion-dpo-softPrompt/INITPROMPTA_photo_of_a_boy_throwing_a_ball_for_a_puppy_NCTX8_EVALGOALboy_throwing_puppy_MLLMinternvl_REWARDqa_score/final_results/    'boy_throwing_puppy puppy_throwing_boy' internvl  ${METRICS}"
#         "output/diffusion-dpo-softPrompt/INITPROMPTA_photo_of_a_puppy_throwing_a_ball_for_a_boy_NCTX8_EVALGOALpuppy_throwing_boy_MLLMinternvl_REWARDqa_score/final_results/   'boy_throwing_puppy puppy_throwing_boy' internvl   ${METRICS}"
#         "output/diffusion-dpo-softPrompt/INITPROMPTA_photo_of_an_astronaut_riding_on_a_horse_NCTX8_EVALGOALastronaut_riding_horse_MLLMinternvl_REWARDqa_score/final_results/  'astronaut_riding_horse horse_riding_astronaut' internvl ${METRICS}"
#         "output/diffusion-dpo-softPrompt/INITPROMPTA_photo_of_a_horse_riding_on_an_astronaut_NCTX8_EVALGOALhorse_riding_astronaut_MLLMinternvl_REWARDqa_score/final_results    'astronaut_riding_horse horse_riding_astronaut' internvl ${METRICS}"
#     )

# PARAMS=(
#     "../../image_score_pairs/init_prompts/${MODEL}/A_photo_of_a_man_feeding_a_horse/   'man_feeding_horse horse_feeding_man' internvl 'qa_score rel_score entropy_score'"
#     "../../image_score_pairs/init_prompts/${MODEL}/A_photo_of_a_horse_feeding_food_to_a_man/   'man_feeding_horse horse_feeding_man' internvl   'qa_score rel_score entropy_score'"
#     "../../image_score_pairs/init_prompts/${MODEL}/A_photo_of_a_mouse_chasing_a_cat/   'cat_chasing_mouse mouse_chasing_cat' internvl 'rel_score entropy_score'"
#     "../../image_score_pairs/init_prompts/${MODEL}/A_photo_of_a_cat_chasing_a_mouse/   'cat_chasing_mouse mouse_chasing_cat' internvl 'rel_score entropy_score'"
#     "../../image_score_pairs/init_prompts/${MODEL}/A_photo_of_a_boy_throwing_a_ball_for_a_puppy/    'boy_throwing_puppy puppy_throwing_boy' internvl  'qa_score rel_score entropy_score'"
#     "../../image_score_pairs/init_prompts/${MODEL}/A_photo_of_a_puppy_throwing_a_ball_for_a_boy/   'boy_throwing_puppy puppy_throwing_boy' internvl   'qa_score rel_score entropy_score'"
#     "../../image_score_pairs/init_prompts/${MODEL}/A_photo_of_an_astronaut_riding_on_a_horse/   'astronaut_riding_horse horse_riding_astronaut' internvl 'qa_score rel_score entropy_score'"
#     "../../image_score_pairs/init_prompts/${MODEL}/A_photo_of_a_horse_riding_on_an_astronaut/   'astronaut_riding_horse horse_riding_astronaut' internvl 'qa_score rel_score entropy_score'"
#     # "../../image_score_pairs/init_prompts/IterComp/A_photo_of_an_astronaut_riding_on_a_horse/   'astronaut_riding_horse horse_riding_astronaut' internvl 'qa_score rel_score entropy_score'"
#     # "../../image_score_pairs/init_prompts/IterComp/A_photo_of_a_horse_riding_on_an_astronaut/   'astronaut_riding_horse horse_riding_astronaut' internvl 'qa_score rel_score entropy_score'"
# #     # "output/diffusion-dpo-softPrompt/INITPROMPTA_photo_of_a_mouse_chasing_a_cat_NCTX8_EVALGOALmouse_chasing_cat_MLLMinternvl/final_results   'cat_chasing_mouse mouse_chasing_cat' internvl 'qa_score rel_score entropy_score'"
# #     # "output/diffusion-dpo-softPrompt/INITPROMPTA_photo_of_a_cat_chasing_a_mouse_NCTX8_EVALGOALcat_chasing_mouse_MLLMinternvl/final_results   'cat_chasing_mouse mouse_chasing_cat' internvl 'qa_score rel_score entropy_score'"
# #     # "output/diffusion-dpo-softPrompt/INITPROMPTA_photo_of_a_cat_chasing_a_mouse_NCTX8_EVALGOALmouse_chasing_cat_MLLMinternvl/final_results   'cat_chasing_mouse mouse_chasing_cat' internvl 'qa_score rel_score entropy_score'"
# #     # "output/diffusion-dpo-softPrompt/INITPROMPTA_photo_of_a_mouse_chasing_a_cat_NCTX8_EVALGOALmouse_chasing_cat_MLLMinternvl/final_results   'cat_chasing_mouse mouse_chasing_cat' gpt4v 'qa_score rel_score entropy_score'"
# )

# PARAMS=(
#     "../../image_score_pairs/init_prompts/${MODEL}/A_photo_of_a_man_feeding_a_horse/   'man_feeding_horse horse_feeding_man' internvl 'qa_score rel_score entropy_score'"
#     "../../image_score_pairs/init_prompts/${MODEL}/A_photo_of_a_horse_feeding_food_to_a_man/   'man_feeding_horse horse_feeding_man' internvl   'qa_score rel_score entropy_score'"
#     "../../image_score_pairs/init_prompts/${MODEL}/A_photo_of_a_mouse_chasing_a_cat/   'cat_chasing_mouse mouse_chasing_cat' internvl 'qa_score rel_score entropy_score'"
#     "../../image_score_pairs/init_prompts/${MODEL}/A_photo_of_a_cat_chasing_a_mouse/   'cat_chasing_mouse mouse_chasing_cat' internvl 'qa_score rel_score entropy_score'"
#     "../../image_score_pairs/init_prompts/${MODEL}/A_photo_of_a_boy_throwing_a_ball_for_a_puppy/    'boy_throwing_puppy puppy_throwing_boy' internvl  'qa_score rel_score entropy_score'"
#     "../../image_score_pairs/init_prompts/${MODEL}/A_photo_of_a_puppy_throwing_a_ball_for_a_boy/   'boy_throwing_puppy puppy_throwing_boy' internvl   'qa_score rel_score entropy_score'"
#     "../../image_score_pairs/init_prompts/${MODEL}/A_photo_of_an_astronaut_riding_on_a_horse/   'astronaut_riding_horse horse_riding_astronaut' internvl 'qa_score rel_score entropy_score'"
#     "../../image_score_pairs/init_prompts/${MODEL}/A_photo_of_a_horse_riding_on_an_astronaut/   'astronaut_riding_horse horse_riding_astronaut' internvl 'qa_score rel_score entropy_score'"
#     # "output/diffusion-dpo-softPrompt/INITPROMPTA_photo_of_a_mouse_chasing_a_cat_NCTX8_EVALGOALmouse_chasing_cat_MLLMinternvl/final_results   'cat_chasing_mouse mouse_chasing_cat' internvl 'qa_score rel_score entropy_score'"
#     # "output/diffusion-dpo-softPrompt/INITPROMPTA_photo_of_a_cat_chasing_a_mouse_NCTX8_EVALGOALcat_chasing_mouse_MLLMinternvl/final_results   'cat_chasing_mouse mouse_chasing_cat' internvl 'qa_score rel_score entropy_score'"
#     # "output/diffusion-dpo-softPrompt/INITPROMPTA_photo_of_a_cat_chasing_a_mouse_NCTX8_EVALGOALmouse_chasing_cat_MLLMinternvl/final_results   'cat_chasing_mouse mouse_chasing_cat' internvl 'qa_score rel_score entropy_score'"
#     # "output/diffusion-dpo-softPrompt/INITPROMPTA_photo_of_a_mouse_chasing_a_cat_NCTX8_EVALGOALmouse_chasing_cat_MLLMinternvl/final_results   'cat_chasing_mouse mouse_chasing_cat' gpt4v 'qa_score rel_score entropy_score'"
# )


# Extract the parameter set for the current job
param="${PARAMS[$SLURM_ARRAY_TASK_ID]}"
eval "set -- $param"  # Split the parameters into $1, $2, $3

# Assign each parameter to a variable
log_dir="$1"
rel_prompt_triplet="$2"
MLLM_evaluator="$3"
metrics="$4"
expanded_flag="$5"    
# Execute the script with the extracted parameters
bash runners/evaluate_single.sh "$log_dir" "$rel_prompt_triplet" "$MLLM_evaluator" "$metrics" "$expanded_flag"

