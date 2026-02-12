#!/bin/bash

# Extract parameters
log_dir="$1"
rel_prompt_triplet="$2"
MLLM_evaluator="$3" 
metrics="$4"
expanded_flag="$5"  # Changed from image_dir_name to expanded_flag


# Echo the parameters for debugging
echo "Parameters received:"
echo "log_dir: $log_dir"    
echo "rel_prompt_triplet: $rel_prompt_triplet"
echo "MLLM_evaluator: $MLLM_evaluator"
echo "metrics: $metrics"
echo "expanded_flag: $expanded_flag"

for seed in 1 
do
    python tools/eval_mllm.py \
        --log_dir $log_dir \
        --rel_prompt_triplet $rel_prompt_triplet \
        --MLLM_evaluator $MLLM_evaluator \
        --metrics $metrics \
        --seed $seed \
        --expanded_images  \
        # --use_VQA_scores \
        # ${expanded_flag:+"$expanded_flag"}  # Only add if expanded_flag is not empty
done
