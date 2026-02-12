#!/bin/bash
#SBATCH --job-name=sdxl_array
#SBATCH --output=logs_iccv_rebuttal/sdxl_train_%A_%a.out
#SBATCH --error=logs_iccv_rebuttal/sdxl_train_%A_%a.err
#SBATCH --time=0:30:00
#SBATCH --cluster=gpu
#SBATCH --partition=a100_nvlink
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-9

# Create logs directory if it doesn't exist
mkdir -p logs

# Define array of relations to evaluate
RELATIONS=(
    "chasing"
    "riding"
    "throwing"
    "holding"
    "following"
    "feeding"
    "pulling"
    "lifting"
    "carrying"
    "kissing"
)

# Get the relation for this array task
target_relation=${RELATIONS[$SLURM_ARRAY_TASK_ID]}

# Set common variables
output_base_dir="ICCV_output"
pretrained_model="stabilityai/stable-diffusion-xl-base-1.0"

echo "Running evaluation for relation: ${target_relation}"
echo "Output Base Dir: ${output_base_dir}"

# Create output directory
output_dir="${output_base_dir}/sdxl_lora"
mkdir -p "${output_dir}"
echo "Output Directory: ${output_dir}"

# Run the evaluation script
python scripts/sdxl/sdxl_lora_intermediate_eval_ablation_intermediates.py \
    --pretrained_model_name_or_path="${pretrained_model}" \
    --output_dir="${output_dir}" \
    --resolution=1024 \
    --rank=32 \
    --target_relation="${target_relation}" \
    --mixed_precision="bf16" \
    --eval_only \
    --resume_from_checkpoint="latest" \
    --gradient_accumulation_steps=4 \
    --gradient_checkpointing

# Check if the evaluation was successful
if [ $? -eq 0 ]; then
    echo "Evaluation completed successfully for ${target_relation}"
else
    echo "Evaluation failed for ${target_relation}"
    exit 1
fi