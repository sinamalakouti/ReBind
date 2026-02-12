#!/bin/bash
#SBATCH --job-name=ghias
#SBATCH --output=rolebench_logs/logs_%A_%a.out
#SBATCH --error=rolebench_logs/logs_%A_%a.err
#SBATCH --time=04:30:00
#SBATCH --cluster=gpu
#SBATCH --partition=a100_nvlink
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --array=0

# Define output directory
output_dir="PATH"

# Define experiment configurations
# Format: "triplet model expand_prompt llm_model"

t2i="itercomp"           # "auraflow2"
N=20
expand_prompt_flag="false"
base_prompt_flag="true"
# bas
# declare -A experiments=(
#     [0]="${t2i} ${N} true ${llm}"
#     [1]="${t2i} ${N} true ${llm}"
#     [2]="${t2i} ${N} true ${llm}"
#     [3]="${t2i} ${N} true ${llm}"
    
# )

# Get the experiment configuration for this array task
# config=(${experiments[$SLURM_ARRAY_TASK_ID]})

# model="${config[0]}"
# num_images="${config[1]:-10}"
# expand_prompt="${config[2]:-false}"  # Default to false if not specified
# llm_model="${config[3]:-}"          # Default to empty if not specified

model_name="${t2i}"
num_images="${N}"
expand_prompt="${expand_prompt_flag}"  # Default to false if not specified
base_prompt="${base_prompt_flag}"

echo "Running inference with parameters:"
echo "Output Directory: ${output_dir}"
echo "Expand Prompt: ${expand_prompt}"
echo "Base Prompt: ${base_prompt}"


cmd="python dataset/generate_rolebench.py \
    --output_dir \"${output_dir}\" \
    --T2I \"${model_name}\" \
    --num_images \"${num_images}\" \
    --seed -1\
    --evaluator_model vqascore_clip-flant5-xxl \
    --metrics qa_score vqascore \
    --evaluate \
    "

# Add expand_prompt and llm_model if specified
if [ "${expand_prompt}" = "true" ]; then
    cmd="${cmd} --expand_prompt"
fi

if [ "${base_prompt}" = "true" ]; then
    cmd="${cmd} --base_prompt"
fi

# Execute the command
eval ${cmd}