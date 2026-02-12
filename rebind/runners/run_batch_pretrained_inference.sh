#!/bin/bash
#SBATCH --job-name=ghias
#SBATCH --output=slurm_logs/pretrained_inference_%A_%a.out
#SBATCH --error=slurm_logs/pretrained_inference_%A_%a.err
#SBATCH --time=02:30:00
#SBATCH --cluster=gpu
#SBATCH --partition=a100_nvlink
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-4     

# Define output directory
output_dir="PATH"

# Define experiment configurations
# Format: "triplet model expand_prompt llm_model"

t2i="dall-e-3"           # "auraflow2"
llm="gpt-4o"   #"llama3_1" #"gpt-4o-mini"
N=20

# declare -A experiments=(
#     [0]="mouse_chasing_cat ${t2i} ${N} true ${llm}"
#     [1]="mouse_chasing_mouse ${t2i} ${N} true ${llm}"
#     [2]="mouse_chasing_boy ${t2i} ${N} true ${llm}"
#     [3]="boy_chasing_mouse ${t2i} ${N} true ${llm}"
#     [4]="mouse_chasing_apple ${t2i} ${N} true ${llm}"
#     [5]="cat_chasing_cat ${t2i} ${N} true ${llm}"
#     [6]="cat_chasing_boy ${t2i} ${N} true ${llm}"
#     [7]="boy_chasing_cat ${t2i} ${N} true ${llm}"
#     [8]="dog_chasing_cat ${t2i} ${N} true ${llm}"
#     [9]="cat_chasing_mouse ${t2i} ${N} true ${llm}"
#     # [10]="women_feeding_baby ${t2i} ${N} true ${llm}"
# )


declare -A experiments=(
    # [0]="kid_riding_astronaut ${t2i} ${N} true ${llm}"
    # [1]="bear_riding_astronaut ${t2i} ${N} true ${llm}"
    # [2]="horse_riding_bear ${t2i} ${N} true ${llm}"
    # [1]="horse_riding_horse ${t2i} ${N} true ${llm}"
    # [2]="horse_riding_zebra ${t2i} ${N} true ${llm}"
    # [3]="zebra_riding_horse ${t2i} ${N} true ${llm}"
    # [4]="horse_riding_motorcycle ${t2i} ${N} true ${llm}"
    # [5]="astronaut_riding_astronaut ${t2i} ${N} true ${llm}"
    # [6]="cat_riding_astronaut ${t2i} ${N} true ${llm}"
    # [7]="boy_chasing_cat ${t2i} ${N} true ${llm}"
    # [8]="astronaut_riding_cat ${t2i} ${N} true ${llm}"
    # [9]="dog_riding_astronaut ${t2i} ${N} true ${llm}"
    # 1-==
    # [10]="women_feeding_baby ${t2i} ${N} true ${llm}"
)

declare -A experiments=(
    [0]="mouse_chasing_cat ${t2i} ${N} true ${llm}"
    [1]="cat_chasing_mouse ${t2i} ${N} true ${llm}"
    [2]="mouse_chasing_boy ${t2i} ${N} true ${llm}"
    [3]="boy_chasing_cat ${t2i} ${N} true ${llm}"
    
)

# declare -A experiments=(
#     [0]="horse_riding_astronaut ${t2i} ${N} true ${llm}"
#     [1]="astronaut_riding_horse ${t2i} ${N} true ${llm}"
#     [2]="mouse_chasing_cat ${t2i} ${N} true ${llm}"
#     [3]="cat_chasing_mouse ${t2i} ${N} true ${llm}"
#     [4]="puppy_throwing_boy ${t2i} ${N} true ${llm}"
#     [5]="boy_throwing_puppy ${t2i} ${N} true ${llm}"
#     [6]="dog_chasing_cat ${t2i} ${N} true ${llm}"
#     [7]="horse_feeding_man ${t2i} ${N} true ${llm}"
#     [8]="man_feeding_horse ${t2i} ${N} true ${llm}"
#     [9]="baby_feeding_women ${t2i} ${N} true ${llm}"
#     [10]="women_feeding_baby ${t2i} ${N} true ${llm}"
# )

# declare -A experiments=(
#     [0]="horse_riding_horse ${t2i} 50 true ${llm}"
#     [1]="astronaut_riding_astronaut ${t2i} 50 true ${llm}"
#     [2]="mouse_chasing_mouse ${t2i} 50 true ${llm}"
#     [3]="cat_chasing_cat ${t2i} 50 true ${llm}"
#     [4]="puppy_throwing_puppy ${t2i} 50 true ${llm}"
#     [5]="boy_throwing_boy ${t2i} 50 true ${llm}"
#     # [6]="cat_chasing_dog ${t2i} 50 true ${llm}"
#     [6]="dog_chasing_dog ${t2i} 50 true ${llm}"
#     [7]="horse_feeding_horse ${t2i} 50 true ${llm}"
#     [8]="man_feeding_man ${t2i} 50 true ${llm}"
#     [9]="baby_feeding_baby ${t2i} 50 true ${llm}"
#     [10]="women_feeding_women ${t2i} 50 true ${llm}"
# )

# Get the experiment configuration for this array task
config=(${experiments[$SLURM_ARRAY_TASK_ID]})
target_triplet="${config[0]}"
model="${config[1]}"
num_images="${config[2]:-10}"
expand_prompt="${config[3]:-false}"  # Default to false if not specified
llm_model="${config[4]:-}"          # Default to empty if not specified


echo "Running inference with parameters:"
echo "Output Directory: ${output_dir}"
echo "Target Triplet: ${target_triplet}"
echo "Model: ${model}"
echo "Expand Prompt: ${expand_prompt}"
echo "LLM Model: ${llm_model}"

# Run the single inference script
bash runners/run_single_pretrained_inference.sh \
    "${target_triplet}" \
    "${model}" \
    "${num_images}" \
    "${expand_prompt}" \
    "${llm_model}" \
    "${output_dir}"

    