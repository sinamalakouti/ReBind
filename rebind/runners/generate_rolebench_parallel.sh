#!/bin/bash
#SBATCH --job-name=t2i_bench
#SBATCH --output=rolebench_logs/logs_%A_%a.out
#SBATCH --error=rolebench_logs/logs_%A_%a.err
#SBATCH --time=04:30:00
#SBATCH --cluster=gpu
#SBATCH --partition=a100_nvlink
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --array=0  # Adjust this based on number of models

# Define output directory
output_dir="PATH"

# Define the T2I models to test
declare -a t2i_models=(
    "dall-e-3"
    "sdxl"
    "sd3"
    "sd3-5"
    "auraflow2"
    "itercomp"
    )

# Common parameters
N=20
expand_prompt_flag="false"
base_prompt_flag="true"

# Get the model for this array task
model_name="${t2i_models[$SLURM_ARRAY_TASK_ID]}"
num_images="${N}"
expand_prompt="${expand_prompt_flag}"
base_prompt="${base_prompt_flag}"

echo "Running inference with parameters:"
echo "Model: ${model_name}"
echo "Output Directory: ${output_dir}"
echo "Number of Images: ${num_images}"
echo "Expand Prompt: ${expand_prompt}"
echo "Base Prompt: ${base_prompt}"

cmd="python dataset/generate_rolebench.py \
    --output_dir \"${output_dir}\" \
    --T2I \"${model_name}\" \
    --num_images \"${num_images}\" \
    --seed -1 \
    --evaluator_model vqascore_clip-flant5-xxl \
    --metrics qa_score vqascore \
    --evaluate"

# Add expand_prompt and base_prompt if specified
if [ "${expand_prompt}" = "true" ]; then
    cmd="${cmd} --expand_prompt"
fi

if [ "${base_prompt}" = "true" ]; then
    cmd="${cmd} --base_prompt"
fi

# Execute the command
eval ${cmd} 