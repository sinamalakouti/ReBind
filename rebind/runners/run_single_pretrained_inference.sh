#!/bin/bash

target_triplet=$1
model_name=$2
num_images=$3
expand_prompt=${4:-"false"}  # Optional third argument, defaults to "false"
llm_model=${5:-""}          # Optional fourth argument, defaults to empty string
output_dir=${6:-"outputs"}  # Optional fifth argument, defaults to "outputs"

echo "Running inference with parameters:"
echo "Target Triplet: ${target_triplet}"
echo "Model Name: ${model_name}"
echo "Expand Prompt: ${expand_prompt}"
echo "LLM Model: ${llm_model}"
echo "Output Directory: ${output_dir}"

# Base command
cmd="python scripts/inference_pretrained.py \
    --triplet \"${target_triplet}\" \
    --output_dir \"${output_dir}\" \
    --T2I \"${model_name}\" \
    --num_images \"${num_images}\" \
    --seed -1"

# Add expand_prompt and llm_model if specified
if [ "${expand_prompt}" = "true" ]; then
    cmd="${cmd} --expand_prompt"
    if [ ! -z "${llm_model}" ]; then
        cmd="${cmd} --llm_model_name \"${llm_model}\""
    fi
fi

# Execute the command
eval ${cmd}


