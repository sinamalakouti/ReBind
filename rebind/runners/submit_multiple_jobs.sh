#!/bin/bash
#SBATCH --job-name=sdxl_array
#SBATCH --output=logs_iccv_rebuttal/sdxl_train_%A_%a.out
#SBATCH --error=logs_iccv_rebuttal/sdxl_train_%A_%a.err
#SBATCH --time=03:30:00
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-1
# Define arrays for triplets and modes
TRIPLETS=(
    # "horse_riding_astronaut"
    # "mouse_chasing_cat"
    # "puppy_throwing_boy"
    # "doll_holding_grandpa"
    # "cow_following_lion"
    "baby_feeding_woman"
    "baby_kissing_mother"
    # "dog_pulling_man"
    # "monkey_lifting_zoo_trainer"
    # "scientist_carrying_fireman"
)

MODES=(
    "sup_only"
    # "contrastive_dpo"
    # "sup_contrastive_dpo"
)

NUM_EXAMPLES_PER_INTERMEDIATE=(
    "full"
    # "1"
    # "3"
    # "5"
    # "10"
    # "15",
    # "20",
    # "25"
    # "30"
)

NUM_INTERMEDIATES_PER_RELATION=(
    "full"
)

# Define intermediate types as arrays to match the Python script's new format
INTERMEDIATE_TYPES=(
    # "active passive"  # This will be passed as --intermediate_types active passive
    "active"         # This will be passed as --intermediate_types passive
    # "passive"          # This will be passed as --intermediate_types active
)

# Initialize experiments array
declare -A experiments

# Create all combinations using nested loops
index=0
for mode in "${MODES[@]}"; do
    for triplet in "${TRIPLETS[@]}"; do
        for num_examples in "${NUM_EXAMPLES_PER_INTERMEDIATE[@]}"; do
            for num_intermediates in "${NUM_INTERMEDIATES_PER_RELATION[@]}"; do
                for int_types in "${INTERMEDIATE_TYPES[@]}"; do
                    experiments[$index]="$triplet $mode $num_examples $num_intermediates $int_types"
                    echo "Created experiment[$index]: $triplet $mode $num_examples $num_intermediates $int_types"
                    ((index++))
                done
            done
        done
    done
done

# Calculate total number of experiments
total_experiments=$index
echo "Total experiments: $total_experiments"

# Get the experiment configuration for this array task
config=(${experiments[$SLURM_ARRAY_TASK_ID]})
target_triplet="${config[0]}"
training_mode="${config[1]}"
num_examples_per_intermediate="${config[2]}"
num_intermediates_per_relation="${config[3]}"
intermediate_types="${config[4]}" # This will be "active passive" or "passive" or "active"
active_weight=1

echo "Running training with parameters:"
echo "Target Triplet: ${target_triplet}"
echo "Training Mode: ${training_mode}"
echo "Num Examples Per Intermediate: ${num_examples_per_intermediate}"
echo "Num Intermediates Per Relation: ${num_intermediates_per_relation}"
echo "Intermediate Types: ${intermediate_types}"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"

# Run the single training script with all parameters
# Note: We're passing intermediate_types as separate arguments
bash runners/train_sdxl.sh \
    "${target_triplet}" \
    "${training_mode}" \
    "${num_examples_per_intermediate}" \
    "${num_intermediates_per_relation}" \
    "${intermediate_types}" \
    "${active_weight}"  # No quotes here to allow word splitting for array arguments 
