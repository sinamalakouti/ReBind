#!/bin/bash
#SBATCH --job-name=eval_t2i
#SBATCH --output=logs/eval_t2i_%A_%a.out
#SBATCH --error=logs/eval_t2i_%A_%a.err
#SBATCH --time=00:30:00
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

# Set T2I model (can be passed as an argument to the script)
T2I_MODEL=${1:-"sdxl"}  # default to sdxl if not specified

echo "Running evaluation for relation: ${target_relation}"
echo "Using T2I model: ${T2I_MODEL}"

# Run the evaluation script
python scripts/ablation_eval_intermediates/eval_sdxl_pretrained_intermediate.py \
    --relation="${target_relation}" \
    --T2I="${T2I_MODEL}" \
    --evaluate

# Check if the evaluation was successful
if [ $? -eq 0 ]; then
    echo "Evaluation completed successfully for ${target_relation}"
else
    echo "Evaluation failed for ${target_relation}"
    exit 1
fi