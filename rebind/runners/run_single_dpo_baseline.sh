#!/bin/bash

# Extract parameters
init_prompt="$1"
eval_goal="$2"
n_questions="$3"
eval_MLLM="$4"
# Echo the parameters for debugging
echo "Parameters received:"
echo "init_prompt: $init_prompt"
echo "eval_goal: $eval_goal"
echo "n_questions: $n_questions"
echo "eval_MLLM: $eval_MLLM"

# Run the softprompt_training.py script using accelerate
accelerate launch scripts/dpo_baseline.py \
  --pretrained_model_name_or_path=stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --output_dir="output/diffusion_dpo_baseline" \
  --mixed_precision="fp16" \
  --dataset_name=kashif/pickascore \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=2  \
  --gradient_checkpointing \
  --use_8bit_adam \
  --learning_rate=1e-5 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=5000 \
  --checkpointing_steps=200 \
  --run_validation --validation_steps=200 \
  --seed="0" \
  --num_train_epochs 5000 \
  --dataset_generator_step 32 \
  --init_prompt="$init_prompt" \
  --VLM_n_questions="$n_questions" \
  --eval_goal="$eval_goal" \
  --MLLM_evaluator="$eval_MLLM" \
  # --eval_only \
  # --resume_from_checkpoint="latest"
  # --add_pos_embeddings  
# 