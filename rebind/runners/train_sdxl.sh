#!/bin/bash

# Get parameters from command line
target_triplet=$1
training_mode=$2
num_examples_per_intermediate=$3
num_intermediates_per_relation=$4
intermediate_types=$5  # This is a space-separated string like "active passive"
active_weight=$6
# Define common variables
output_base_dir="SDXL_rebuttal" # ICCV_output
pretrained_model="stabilityai/stable-diffusion-xl-base-1.0"
data_dir="data/dall-e-3"

echo "Running training with parameters:"
echo "Target Triplet: ${target_triplet}"
echo "Training Mode: ${training_mode}"
echo "Num Examples Per Intermediate: ${num_examples_per_intermediate}"
echo "Num Intermediates Per Relation: ${num_intermediates_per_relation}"
echo "Intermediate Types: ${intermediate_types}"
echo "Output Base Dir: ${output_base_dir}"
echo "Data Dir: ${data_dir}"

# Create output directory
# output_dir="${output_base_dir}/sdxl_lora_${training_mode}_${target_triplet}"
output_dir="${output_base_dir}/sdxl_lora"
mkdir -p "${output_dir}"
echo "Output Directory: ${output_dir}"

# Run the training script with all necessary arguments
python scripts/sdxl/rebind_lora_sdxl.py \
    --pretrained_model_name_or_path="${pretrained_model}" \
    --data_dir="${data_dir}" \
    --output_dir="${output_dir}" \
    --resolution=1024 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=1000 \
    --learning_rate=1e-4 \
    --rank=32 \
    --validation_epochs=15 \
    --validation_prompt="" \
    --num_validation_images=1 \
    --gradient_checkpointing \
    --caption_column="caption" \
    --target_triplets="${target_triplet}" \
    --mixed_precision="bf16" \
    --training_mode="${training_mode}" \
    --weighted_sup \
    --active_weight="${active_weight}" \
    --num_examples_per_intermediate="${num_examples_per_intermediate}" \
    --num_intermediates_per_relation="${num_intermediates_per_relation}" \
    --intermediate_types ${intermediate_types} \
    # --eval_only \ 
    # --resume_from_checkpoint="latest"
    # \
    # --use_freq_rare

# Check if the training was successful
if [ $? -eq 0 ]; then
    echo "Training completed successfully for ${target_triplet} with mode ${training_mode}"
else
    echo "Training failed for ${target_triplet} with mode ${training_mode}"
    exit 1
fi
    # --training_mode="sup_only"
    # --experiment_name="sdx_sup_finetune_only_basline"
    # --eval_only \
    # --resume_from_checkpoint="output/sdxl_lora_mouse_chasing_mouse/mouse_chasing_cat/checkpoint-600"
    # --resume_from_checkpoint="latest"
    # --train_text_encoder
    # --mixed_precision="fp16" \
    # --use_8bit_adam \