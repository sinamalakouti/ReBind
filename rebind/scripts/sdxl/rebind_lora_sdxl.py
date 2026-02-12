#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Stable Diffusion XL for text2image with support for LoRA."""

import argparse
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import datasets
import diffusers
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    DistributedType,
    ProjectConfiguration,
    set_seed,
)
from configs.config import load_config, merge_configs
from dataset.paths import get_data_dir
from dataset.rolebench import (
    get_evaluation_triplets,
    parse_action_triplet,
    rolebench_data,
    reverse_triplet,
)
from datasets import load_dataset
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_snr,
)
from diffusers.utils import (
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from eval.eval import evaluate
from eval.evaluators import get_evaluator_class
from huggingface_hub import create_repo, upload_folder
from packaging import version
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from prompts.prompt_utils import get_base_prompt_from_action_triplet
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

if is_wandb_available():
    import wandb

logger = get_logger(__name__)
if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False


def generate_test_triplets():
    test_triplets = {}
    for relation, data in rolebench_data.items():
        rare_triplet = data["rare"]
        frequent_triplet = data["frequent"]
        test_triplets[rare_triplet] = [rare_triplet, frequent_triplet]
    return test_triplets


TEST_TRIPLETS = generate_test_triplets()
VALIDATION_TRIPLETS = generate_test_triplets()


EVALUATION_STEPS = [5, 200, 400, 600, 800]


def save_model_card(
    repo_id: str,
    images: list = None,
    base_model: str = None,
    dataset_name: str = None,
    train_text_encoder: bool = False,
    repo_folder: str = None,
    vae_path: str = None,
):
    img_str = ""
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            img_str += f"![img_{i}](./image_{i}.png)\n"

    model_description = f"""
# LoRA text2image fine-tuning - {repo_id}

These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. \n
{img_str}

LoRA for the text encoder was enabled: {train_text_encoder}.

Special VAE used for training: {vae_path}.
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion-xl",
        "stable-diffusion-xl-diffusers",
        "text-to-image",
        "diffusers",
        "diffusers-training",
        "lora",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def log_validation(
    pipeline,
    args,
    accelerator,
    epoch,
    is_final_validation=False,
    step=None,
    num_validation_images=1,
    validation_triplets=None,
    run_evaluation=False,
):
    images = []
    prompts = []
    assert validation_triplets is not None, "validation_triplets must be provided"
    for validation_triplet in validation_triplets:
        validation_prompt = get_base_prompt_from_action_triplet(validation_triplet)
        logger.info(
            f"step: {step} , epoch: {epoch} || Running validation... \n Generating {num_validation_images} images with prompt:"
            f" {validation_prompt}."
        )
        pipeline = pipeline.to(accelerator.device)
        pipeline.set_progress_bar_config(disable=True)

        # run inference
        generator = (
            torch.Generator(device=accelerator.device).manual_seed(args.seed)
            if args.seed
            else None
        )

        pipeline_args = {"prompt": validation_prompt}
        # if torch.backends.mps.is_available():
        #     autocast_ctx = nullcontext()
        # else:
        #     autocast_ctx = torch.autocast(accelerator.device.type)

        # with autocast_ctx:
        # print("HIIIIII prompt   ", validation_prompt)

        # Create directory for this epoch and validation triplet
        if run_evaluation:
            if is_final_validation:
                save_dir = os.path.join(
                    args.output_dir, "validation", f"epoch_{epoch}", validation_triplet
                )
            else:
                save_dir = os.path.join(
                    args.output_dir, "validation", f"step_{step}", validation_triplet
                )
            image_dir = os.path.join(save_dir, "images")
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(image_dir, exist_ok=True)

        # Generate and optionally save images
        for i in range(num_validation_images):
            image = pipeline(**pipeline_args, generator=generator).images[0]
            images.append(image)
            prompts.append(validation_prompt)

            # Save image if it's a validation epoch
            if run_evaluation:
                image_path = os.path.join(image_dir, f"image_{i}.png")
                image.save(image_path)

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {prompts[i]}")
                        for i, image in enumerate(images)
                    ]
                }
            )


def evaluate_images(
    evaluator,
    args,
    cfg,
    accelerator,
    epoch,
    step=None,
    validation_triplets=None,
    is_final_validation=False,
):
    # offload pipeline :)
    results = {}

    for validation_triplet in validation_triplets:
        if is_final_validation:
            save_dir = os.path.join(
                args.output_dir, "validation", f"epoch_{epoch}", validation_triplet
            )
        else:
            save_dir = os.path.join(
                args.output_dir, "validation", f"step_{step}", validation_triplet
            )
        image_dir = os.path.join(save_dir, "images")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)
        num_validation_images = len(os.listdir(image_dir))

        logger.info(
            f"step: {step} , Running validation... \n evaluating {num_validation_images} images with prompt:"
            f" {validation_triplet}."
        )

        for metric in cfg["eval"]["metrics"]:
            for eval_triplet in get_evaluation_triplets(validation_triplet):
                scores, _, _ = evaluate(
                    image_dir=image_dir,
                    eval_triplet=eval_triplet,
                    metric=metric,
                    evaluator=evaluator,
                    output_dir=Path(save_dir) / "results",
                    cfg=cfg,
                )

                # Convert numpy values to Python scalars for logging
                qa_values = list(scores.values())
                logger.info(
                    f"RESULTS for METRIC {metric} for triplet {validation_triplet}/{eval_triplet}: {qa_values}"
                )

                # Create metric path prefix for cleaner code
                metric_path = f"{validation_triplet}/{eval_triplet}/{metric}"
                # Store results with numpy values converted to Python scalars
                results.update(
                    {
                        f"{metric_path}/mean_score": float(np.mean(qa_values)),
                        f"{metric_path}/min_score": float(np.min(qa_values)),
                        f"{metric_path}/max_score": float(np.max(qa_values)),
                    }
                )

    # Log all results at once, after all triplets and metrics are processed
    logger.info(f"EVALUATION RESULTS AT STEP: {step}")
    logger.info("=" * 50)
    for key, value in results.items():
        logger.info(f"{key}: {value}")
    logger.info("=" * 50)

    accelerator.log(results, step=step)


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Fine-tune SDXL with LoRA for relation understanding."
    )
    parser.add_argument(
        "--weighted_sup",
        action="store_true",
        default=False,
        help="Whether to use weighted sup for ImageDPO.",
    )
    parser.add_argument(
        "--active_weight",
        type=float,
        default=2,
        help="Weight for active triplet for ImageDPO.",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        default=False,
        help="Whether to use EMA for ImageDPO.",
    )
    parser.add_argument(
        "--use_ema_ref",
        action="store_true",
        default=False,
        help="Whether to use EMA for reference model for ImageDPO.",
    )

    parser.add_argument(
        "--disable_ref_model",
        action="store_true",
        default=False,
        help="Whether to add reference model for ImageDPO.",
    )
    parser.add_argument(
        "--training_mode",
        default="sup_contrastive_dpo",
        help="training_mode",
    )
    parser.add_argument(
        "--input_is_expanded",
        action="store_true",
        default=False,
        help="Whether the input images are expanded or not.",
    )
    parser.add_argument(
        "--inference_use_expanded",
        action="store_true",
        default=False,
        help="Whether to use freq rare or not.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--vae_encode_batch_size",
        type=int,
        default=8,
        help="Batch size to use for VAE encoding of the images for efficient processing.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing an image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=200,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--enable_npu_flash_attention",
        action="store_true",
        help="Whether or not to use npu flash attention.",
    )
    parser.add_argument(
        "--noise_offset", type=float, default=0, help="The scale of noise offset."
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--debug_loss",
        action="store_true",
        help="debug loss for each image, if filenames are available in the dataset",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the training images",
    )
    parser.add_argument(
        "--target_triplets",
        type=str,
        nargs="+",  # This allows multiple triplets to be passed
        default=["mouse_chasing_cat"],
        help="List of target triplets to train on, e.g., 'mouse_chasing_cat horse_riding_astronaut'",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only evaluate the model on the validation set",
    )

    parser.add_argument(
        "--use_freq_rare",
        action="store_true",
        help="when set uses ferquent and rare images instead of intermediates  to train the model",
    )
    parser.add_argument("--num_examples_per_intermediate", type=str, default="full")
    parser.add_argument("--num_intermediates_per_relation", type=str, default="full")
    parser.add_argument(
        "--intermediate_types",
        type=str,
        nargs="+",  # This allows multiple values to be passed
        default=["active", "passive"],
        help="Types of intermediates to use (e.g., active, passive)",
    )

    # parser.add_argument(
    #     "--experiment_name",
    #     type=str,

    #     default="sdxl_active_passive_dpo_contrastive_finetuing_base_mouse_chasing_cat",
    #     help="Name of the experiment",
    # )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args


def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
            return_dict=False,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds[-1][-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


def get_expanded_prompt(csv_path, image_name):
    """Get expanded prompt for a given image from CSV file"""
    df = pd.read_csv(csv_path)
    # Find the matching row for the image name
    matching_row = df[df["image_name"] == image_name]
    if not matching_row.empty:
        return matching_row["expanded_prompt"].iloc[0]
    return None


def get_full_dataset_size(cfg):
    """Calculate dataset size when using all examples"""
    # Save current N
    original_N = cfg["data"]["num_examples_per_intermediate"]

    # Temporarily set to "full"
    cfg["data"]["num_examples_per_intermediate"] = "full"

    # Get full dataset size
    full_data = create_intermediate_dpo_dataset(data_dir, target_triplet, cfg)
    full_size = len(full_data["image"])

    # Restore original N
    cfg["data"]["num_examples_per_intermediate"] = original_N

    return full_size


def load_triplet_data(
    triplet, image_type="expanded_images", cfg=None, prune=False, threshold=0.7
):
    """Load images and prompts for a single triplet.

    Args:
        data_dir: Base directory containing all data
        triplet: Triplet string (e.g., "mouse_chasing_boy")

    Returns:
        tuple: (images, captions, expanded_captions)
    """
    triplet_dir, prompt_dir, image_dir = get_data_dir(
        triplet, cfg["model"]["base_t2i"], cfg["model"]["llm_model"]
    )  # propmt dir is needed for getting eval propmt?
    prompt_mapping_csv = triplet_dir / "expanded_prompt_mapping.csv"
    if image_type == "expanded_images":
        image_dir = image_dir / "expanded_images"
    else:
        image_dir = image_dir / "images"

    assert triplet_dir.exists(), f"Triplet directory not found: {triplet_dir}"
    assert prompt_dir.exists(), f"Prompt directory not found: {prompt_dir}"
    assert image_dir.exists(), f"Image directory not found: {image_dir}"
    assert (
        prompt_mapping_csv.exists()
    ), f"Prompt mapping CSV not found: {prompt_mapping_csv}"

    images = []
    captions = []
    expanded_captions = []
    score = "vqascore"
    if prune:
        matching_eval_scores = get_eval_scores(
            triplet,
            triplet,
            cfg["model"]["base_t2i"],
            "VQAScore_clip-flant5-xxl",
            "vqascore",
        )
        unmatching_eval_scores = get_eval_scores(
            triplet,
            reverse_triplet(triplet),
            cfg["model"]["base_t2i"],
            "VQAScore_clip-flant5-xxl",
            "vqascore",
        )
    for img_path in image_dir.glob("*.png"):
        image_name = img_path.name
        if prune:
            # print("eval_scores is :  ", eval_scores)
            matching_score = matching_eval_scores[
                matching_eval_scores["image_num"] == image_name
            ]["total"].values[0]
            unmatching_score = unmatching_eval_scores[
                unmatching_eval_scores["image_num"] == image_name
            ]["total"].values[0]
            diff_score = matching_score - unmatching_score
            if diff_score < 0:
                continue
        images.append(Image.open(img_path).convert("RGB"))
        captions.append(get_base_prompt_from_action_triplet(triplet))

        expanded_prompt = get_expanded_prompt(prompt_mapping_csv, img_path.name)
        expanded_captions.append(expanded_prompt or "")

        if not expanded_prompt:
            logger.warning(f"No expanded prompt found for {img_path.name}")

    return images, captions, expanded_captions


def get_eval_scores(
    target_triplet,
    eval_triplet,
    t2i_base,
    evaluator_name,
    metric,
    input_is_expanded=True,
):
    triplet_dir, _, _ = get_data_dir(target_triplet, t2i_base, LLM="gpt-4o")
    results_dir_name = "expanded_results" if input_is_expanded else "results"
    eval_dir = triplet_dir / t2i_base / results_dir_name
    metric_name = metric if metric == "qa_score" else "VQAscore"
    eval_dir = eval_dir / metric_name / evaluator_name
    try:
        if metric == "qa_score":
            eval_scores = pd.read_csv(
                eval_dir / f"qa_scores_{evaluator_name}_triplet_{eval_triplet}.csv"
            )
        elif metric.lower() == "vqascore":
            eval_scores = pd.read_csv(
                eval_dir / f"VQA_scores_{evaluator_name}_triplet_{eval_triplet}.csv"
            )
        else:
            raise ValueError(f"Invalid metric: {metric}")
    except:
        logger.warning(
            f"Didn't find eval scores for metrics {metric} for triplet {target_triplet} at eval dir {eval_dir}. Evaluating now..."
        )
        triplet_dir, _, image_dir = get_data_dir(target_triplet, t2i_base, LLM="gpt-4o")
        results_dir_name = "expanded_results" if input_is_expanded else "results"
        save_dir = triplet_dir / t2i_base / results_dir_name

        evaluator = get_evaluator_class(evaluator_name)
        image_dir = image_dir / "expanded_images"
        eval_scores, _, _ = evaluate(
            image_dir=image_dir,
            eval_triplet=eval_triplet,
            metric=metric,
            evaluator=evaluator,
            output_dir=save_dir,
            cfg=cfg,
        )

        # Convert dict to DataFrame if it's a dictionary
        if isinstance(eval_scores, dict):
            # Convert dictionary to DataFrame
            eval_scores = pd.DataFrame(
                {
                    "image_num": list(eval_scores.keys()),
                    "total": list(eval_scores.values()),
                }
            )

        # Now eval_scores is a DataFrame with 'image_num' and 'total' columns
    return eval_scores


def create_intermediate_dpo_dataset(data_dir: str, target_triplet: str, cfg):
    """Create both DPO and fine-tuning datasets.

    Loads intermediates from intermediate.json file which looks like:
    {
        "cow_following_lion": {
            "passive": [
                {"triplet": "tiger_following_lion", "type": "frequent"},
                {"triplet": "sheep_following_cow", "type": "frequent"}
            ],
            "active": [
                {"triplet": "cow_following_person", "type": "rare"},
                {"triplet": "cow_following_lamb", "type": "frequent"}
            ]
        }
    }
    """
    import json

    # Load intermediates from JSON
    _, prompt_dir, _ = get_data_dir(target_triplet, LLM="gpt-4o", T2I="dall-e-3")
    with open(os.path.join(prompt_dir, "intermediate.json"), "r") as f:
        intermediates_data = json.load(f)

    # Get intermediates for target triplet
    triplet_intermediates = intermediates_data[target_triplet]
    triplet_data = {}

    # Load all data
    for intermediate_type in ["active", "passive"]:
        # Get all intermediates for this type
        intermediates = triplet_intermediates[intermediate_type]
        if cfg["data"]["num_intermediates_per_relation"] != "full":
            intermediates = intermediates[
                : int(cfg["data"]["num_intermediates_per_relation"])
            ]

        for intermediate_info in intermediates:
            intermediate = intermediate_info["triplet"]
            is_rare = intermediate_info["type"] == "rare"

            # Get contrast pair using reverse_triplet
            contrast_pair = reverse_triplet(intermediate)

            # Load intermediate
            images, captions, expanded = load_triplet_data(
                intermediate, cfg=cfg, prune=True
            )
            triplet_data[intermediate] = {
                "images": images,
                "captions": captions,
                "expanded": expanded,
            }

            # Load its contrast pair
            contrast_images, contrast_captions, contrast_expanded = load_triplet_data(
                contrast_pair, cfg=cfg, prune=False
            )
            triplet_data[contrast_pair] = {
                "images": contrast_images,
                "captions": contrast_captions,
                "expanded": contrast_expanded,
            }

    # Create datasets
    preference_data = {
        "image_winner": [],
        "image_loser": [],
        "caption": [],
        "expanded_caption": [],
        "triplet": [],
        "caption_loser": [],
        "expanded_caption_loser": [],
        "label": [],
        "triplet_loser": [],
    }
    active_data = {"image": [], "caption": [], "expanded_caption": [], "triplet": []}
    passive_data = {"image": [], "caption": [], "expanded_caption": [], "triplet": []}

    # TODO: sis this right? or should I again ensure same number of examples? this could be super different due to pruning!

    if cfg["data"]["num_examples_per_intermediate"] != "full":
        N = int(cfg["data"]["num_examples_per_intermediate"])
    else:
        N = None
    # Process active and passive intermediates
    # print("intermediate_types: ", cfg["data"]["intermediate_types"])
    print(
        "cfg['data']['num_intermediates_per_relation']: ",
        cfg["data"]["num_intermediates_per_relation"],
    )
    print(
        "cfg['data']['num_examples_per_intermediate']: ",
        cfg["data"]["num_examples_per_intermediate"],
    )
    for intermediate_type in ["active", "passive"]:
        intermediates = triplet_intermediates[intermediate_type]

        for intermediate_info in intermediates:
            intermediate = intermediate_info["triplet"]
            is_rare = intermediate_info["type"] == "rare"
            contrast_pair = reverse_triplet(intermediate)

            n_pairs = min(
                len(triplet_data[intermediate]["images"]),
                len(triplet_data[contrast_pair]["images"]),
            )
            if N is None:
                N_per_relation = len(triplet_data[intermediate]["images"])
            else:
                N_per_relation = min(N, len(triplet_data[intermediate]["images"]))

            if is_rare:
                # Add rare to winners
                preference_data["image_winner"].extend(
                    triplet_data[intermediate]["images"][:n_pairs]
                )
                preference_data["caption"].extend(
                    triplet_data[intermediate]["expanded"][:n_pairs]
                )
                preference_data["expanded_caption"].extend(
                    triplet_data[intermediate]["expanded"][:n_pairs]
                )
                preference_data["triplet"].extend([intermediate] * n_pairs)

                # Add contrast to losers
                preference_data["image_loser"].extend(
                    triplet_data[contrast_pair]["images"][:n_pairs]
                )
                preference_data["caption_loser"].extend(
                    triplet_data[contrast_pair]["captions"][:n_pairs]
                )
                preference_data["expanded_caption_loser"].extend(
                    triplet_data[contrast_pair]["expanded"][:n_pairs]
                )
                preference_data["triplet_loser"].extend([contrast_pair] * n_pairs)

                preference_data["label"].extend([1] * n_pairs)
            else:
                # Add frequent to losers
                preference_data["image_loser"].extend(
                    triplet_data[intermediate]["images"][:n_pairs]
                )
                preference_data["caption_loser"].extend(
                    triplet_data[intermediate]["captions"][:n_pairs]
                )
                preference_data["expanded_caption_loser"].extend(
                    triplet_data[intermediate]["expanded"][:n_pairs]
                )
                preference_data["triplet_loser"].extend([intermediate] * n_pairs)

                # Add contrast to winners
                preference_data["image_winner"].extend(
                    triplet_data[contrast_pair]["images"][:n_pairs]
                )
                preference_data["caption"].extend(
                    triplet_data[contrast_pair]["captions"][:n_pairs]
                )
                preference_data["expanded_caption"].extend(
                    triplet_data[contrast_pair]["expanded"][:n_pairs]
                )
                preference_data["triplet"].extend([contrast_pair] * n_pairs)

                preference_data["label"].extend([1] * n_pairs)

            # Add to active/passive datasets
            print("FOR PASSIVE/ACTIVE DATASETS")
            print("intermediate_type: ", intermediate_type)
            print("N: ", N)

            logger.info(
                f"N: {N}, num_active_before_filtering: {len(active_data['image'])}, num_passive_before_filtering: {len(passive_data['image'])}"
            )
            logger.info(
                f"N: {N}, num_active_before_filtering: {len(active_data['caption'])}, num_passive_before_filtering: {len(passive_data['image'])}"
            )

            if intermediate_type == "active":
                active_data["image"].extend(
                    triplet_data[intermediate]["images"][:N_per_relation]
                )
                active_data["caption"].extend(
                    triplet_data[intermediate]["captions"][:N_per_relation]
                )
                active_data["expanded_caption"].extend(
                    triplet_data[intermediate]["expanded"][:N_per_relation]
                )
                active_data["triplet"].extend(
                    [intermediate]
                    * len(triplet_data[intermediate]["images"][:N_per_relation])
                )
            else:  # passive
                passive_data["image"].extend(
                    triplet_data[intermediate]["images"][:N_per_relation]
                )
                passive_data["caption"].extend(
                    triplet_data[intermediate]["captions"][:N_per_relation]
                )
                passive_data["expanded_caption"].extend(
                    triplet_data[intermediate]["expanded"][:N_per_relation]
                )
                passive_data["triplet"].extend(
                    [intermediate]
                    * len(triplet_data[intermediate]["images"][:N_per_relation])
                )

    print(len(preference_data["caption"]))
    print(len(preference_data["image_winner"]))

    print("preference data stats")
    for key in preference_data.keys():
        print(f"  key: {key}  len: {len(preference_data[key])}")

    print("active data stats")
    for key in active_data.keys():
        print(f"  key: {key}  len: {len(active_data[key])}")

    print("passive data stats")
    for key in passive_data.keys():
        print(f"  key: {key}  len: {len(passive_data[key])}")

    # Create HuggingFace datasets
    return (
        datasets.Dataset.from_dict(preference_data),
        datasets.Dataset.from_dict(active_data),
        datasets.Dataset.from_dict(passive_data),
    )


def create_freq_rare_dpo_dataset(data_dir: str, target_triplet: str, cfg, args):
    """Create DPO dataset using frequent/rare pairs instead of intermediates."""
    assert (
        not args.input_is_expanded
    ), "Only non-expanded images are supported for freq/rare DPO"

    # Initialize datasets with same structure as intermediate version
    preference_data = {
        "image_winner": [],
        "image_loser": [],
        "caption": [],
        "expanded_caption": [],
        "triplet": [],
        "caption_loser": [],
        "expanded_caption_loser": [],
        "label": [],
        "triplet_loser": [],
    }
    active_data = {"image": [], "caption": [], "expanded_caption": [], "triplet": []}
    passive_data = {"image": [], "caption": [], "expanded_caption": [], "triplet": []}

    # Get rare and frequent triplets
    _, relation, _ = parse_action_triplet(target_triplet)
    rare_triplet = target_triplet
    frequent_triplet = rolebench_data[relation]["frequent"]

    # Load data using the same helper function as intermediate version
    rare_images, rare_captions, rare_expanded = load_triplet_data(
        rare_triplet,
        cfg=cfg,
        prune=False,
        image_type="expanded_images" if args.input_is_expanded else "images",
    )
    freq_images, freq_captions, freq_expanded = load_triplet_data(
        frequent_triplet,
        cfg=cfg,
        prune=False,
        image_type="expanded_images" if args.input_is_expanded else "images",
    )

    # Get number of pairs to use for preferece data as we need same number winners/losers (i.e. rare/frequent)
    n_pairs = min(len(rare_images), len(freq_images))
    # TODO: sis this right? or should I again ensure same number of examples? this could be super different due to pruning!

    if cfg["data"]["num_examples_per_intermediate"] != "full":
        N = int(cfg["data"]["num_examples_per_intermediate"])
    else:
        N = None

    # Add rare triplet to winners (following same pattern as intermediate version)
    preference_data["image_winner"].extend(rare_images[:n_pairs])
    preference_data["caption"].extend(rare_captions[:n_pairs])
    preference_data["expanded_caption"].extend(rare_expanded[:n_pairs])
    preference_data["triplet"].extend([rare_triplet] * n_pairs)

    # Add frequent triplet to losers
    preference_data["image_loser"].extend(freq_images[:n_pairs])
    preference_data["caption_loser"].extend(freq_captions[:n_pairs])
    preference_data["expanded_caption_loser"].extend(freq_expanded[:n_pairs])
    preference_data["triplet_loser"].extend([frequent_triplet] * n_pairs)

    preference_data["label"].extend([1] * n_pairs)

    # Add to active/passive datasets (using rare triplet for both)
    active_data["image"].extend(rare_images)
    active_data["caption"].extend(rare_captions)
    active_data["expanded_caption"].extend(rare_expanded)
    active_data["triplet"].extend([rare_triplet] * len(rare_images))

    passive_data["image"].extend(rare_images)
    passive_data["caption"].extend(rare_captions)
    passive_data["expanded_caption"].extend(rare_expanded)
    passive_data["triplet"].extend([rare_triplet] * len(rare_images))

    # Print stats (same as intermediate version)
    print("preference data stats")
    for key in preference_data.keys():
        print(f"  key: {key}  len: {len(preference_data[key])}")

    print("\nactive data stats")
    for key in active_data.keys():
        print(f"  key: {key}  len: {len(active_data[key])}")

    print("\npassive data stats")
    for key in passive_data.keys():
        print(f"  key: {key}  len: {len(passive_data[key])}")

    # Return HuggingFace datasets (same as intermediate version)
    return (
        datasets.Dataset.from_dict(preference_data),
        datasets.Dataset.from_dict(active_data),
        datasets.Dataset.from_dict(passive_data),
    )


#  methods for diffusion proceess
def encode_vae_latents(vae, pixel_values, vae_batch_size):
    """Encode images to latent space using VAE with batching."""
    latents = []
    for i in range(0, pixel_values.shape[0], vae_batch_size):
        latents.append(
            vae.encode(pixel_values[i : i + vae_batch_size]).latent_dist.sample()
        )
    latents = torch.cat(latents, dim=0)
    return latents * vae.config.scaling_factor


# time ids
def compute_time_ids(original_size, crops_coords_top_left, accelerator, weight_dtype):
    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
    target_size = (args.resolution, args.resolution)
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids])
    add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
    return add_time_ids


@torch.no_grad()
def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


@torch.no_grad()
def exp_ramp_up(initial_value, target_value, max_step, current_step):
    """
    Calculate exponential ramp-up value between initial and target values.

    Args:
        initial_value (float): Starting value
        target_value (float): Final target value
        num_steps (int): Total number of steps for ramp-up
        current_step (int): Current step

    Returns:
        float: Current value based on exponential ramp-up
    """
    if current_step >= max_step:
        return target_value

    # Ensure current_step doesn't exceed num_steps
    current_step = min(current_step, max_step)

    # Calculate exponential factor
    alpha = current_step / max_step
    # Use exponential curve: e^(ax) - 1 / e^a - 1
    exp_factor = (math.exp(3 * alpha) - 1) / (math.exp(3) - 1)

    # Interpolate between initial and target value
    current_value = initial_value + (target_value - initial_value) * exp_factor

    return current_value


def train_step_active_passive(
    batch,
    vae,
    unet,
    noise_scheduler,
    text_encoder_one,
    text_encoder_two,
    tokenizer_one,
    tokenizer_two,
    accelerator,
    global_step,
    weight_dtype,
    args,
    cfg,
):

    n_active_batch = len(batch["pixel_values"]) // 2
    # Convert images to latent space
    if args.pretrained_vae_model_name_or_path is not None:
        pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
    else:
        pixel_values = batch["pixel_values"]

    # In the main training loop:
    model_input = encode_vae_latents(
        vae, pixel_values, vae_batch_size=args.vae_encode_batch_size
    )

    if args.pretrained_vae_model_name_or_path is None:
        model_input = model_input.to(weight_dtype)

    # Sample noise that we'll add to the latents
    noise = torch.randn_like(model_input)

    if args.noise_offset:
        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
        noise += args.noise_offset * torch.randn(
            (model_input.shape[0], model_input.shape[1], 1, 1),
            device=model_input.device,
        )

    bsz = model_input.shape[0]

    # Sample a random timestep for each image
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (bsz,),
        device=model_input.device,
    )

    timesteps = timesteps.long()

    # Add noise to the model input according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

    add_time_ids = torch.cat(
        [
            compute_time_ids(s, c, accelerator, weight_dtype)
            for s, c in zip(batch["original_sizes"], batch["crop_top_lefts"])
        ]
    )

    # Predict the noise residual
    rand_num = random.random()
    if rand_num < 0.05:
        empty_prompts = [""] * (len(batch["input_ids_one"]))
        unet_added_conditions = {"time_ids": add_time_ids}
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders=[text_encoder_one, text_encoder_two],
            tokenizers=[tokenizer_one, tokenizer_two],
            prompt=empty_prompts,
        )
    else:
        unet_added_conditions = {"time_ids": add_time_ids}
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders=[text_encoder_one, text_encoder_two],
            tokenizers=None,
            prompt=None,
            text_input_ids_list=[
                batch["input_ids_one"],
                batch["input_ids_two"],
            ],
        )

    unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})

    model_pred = unet(
        noisy_model_input,
        timesteps,
        prompt_embeds,
        added_cond_kwargs=unet_added_conditions,
        return_dict=False,
    )[0]

    # Get the target for loss depending on the prediction type
    if args.prediction_type is not None:
        # set prediction_type of scheduler if defined
        noise_scheduler.register_to_config(prediction_type=args.prediction_type)

    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(model_input, noise, timesteps)
    else:
        raise ValueError(
            f"Unknown prediction type {noise_scheduler.config.prediction_type}"
        )

    if args.snr_gamma is None:
        # Compute per-example loss
        sup_loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        # Average across all dimensions except batch
        sup_loss = sup_loss.mean(dim=list(range(1, len(sup_loss.shape))))

        # get active & passive losses for logging purposes
        if "active" in cfg["data"]["intermediate_types"]:
            active_loss = sup_loss[:n_active_batch].mean()
        else:
            active_loss = sup_loss[:n_active_batch].mean() * 0.0
        if "passive" in cfg["data"]["intermediate_types"]:
            passive_loss = sup_loss[n_active_batch:].mean()
        else:
            passive_loss = sup_loss[n_active_batch:].mean() * 0.0

        # Apply weights per example and take mean

        sup_loss = (sup_loss * batch["weights"]).mean()

    else:
        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
        # This is discussed in Section 4.2 of the same paper.
        snr = compute_snr(noise_scheduler, timesteps)

        mse_loss_weights = torch.stack(
            [snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1
        ).min(dim=1)[0]

        if noise_scheduler.config.prediction_type == "epsilon":
            mse_loss_weights = mse_loss_weights / snr

        elif noise_scheduler.config.prediction_type == "v_prediction":
            mse_loss_weights = mse_loss_weights / (snr + 1)

        # Compute per-example loss
        sup_loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        # Average across all dimensions except batch
        sup_loss = sup_loss.mean(dim=list(range(1, len(sup_loss.shape))))

        # get active & passive losses for logging purposes
        active_loss = sup_loss[:n_active_batch].mean()
        passive_loss = sup_loss[n_active_batch:].mean()
        # Apply both SNR weights and our active/passive weights
        sup_loss = sup_loss * batch["weights"]
        # Take final mean
        sup_loss = sup_loss.mean() * mse_loss_weights

    if args.debug_loss and "filenames" in batch:
        for fname in batch["filenames"]:
            accelerator.log({"sup_loss_for_" + fname: sup_loss}, step=global_step)
    # Gather the losses across all processes for logging (if we use distributed training).
    avg_sup_loss = accelerator.gather(sup_loss.repeat(args.train_batch_size)).mean()

    loss_logs = {
        "active_loss": active_loss.detach().item(),
        "passive_loss": passive_loss.detach().item(),
        "sup_loss": sup_loss.detach().item(),
        "avg_sup_loss": avg_sup_loss.detach().item(),
    }
    losses = {"sup_loss": sup_loss}
    return losses, loss_logs


def train_step_contrastive_dpo(
    batch,
    vae,
    unet,
    ema_ref_unet,
    noise_scheduler,
    text_encoder_one,
    text_encoder_two,
    accelerator,
    global_step,
    weight_dtype,
    args,
    cfg,
):

    # Convert images to latent space
    if args.pretrained_vae_model_name_or_path is not None:
        batch["pixel_values"] = batch["pixel_values"].to(dtype=weight_dtype)

    preference_model_input = encode_vae_latents(
        vae,
        torch.cat(batch["pixel_values"].chunk(2, dim=1)),
        vae_batch_size=args.vae_encode_batch_size,
    )

    if args.pretrained_vae_model_name_or_path is None:
        preference_model_input = preference_model_input.to(weight_dtype)

    # Sample noise that we'll add to the latents

    preference_noise = (
        torch.randn_like(preference_model_input).chunk(2)[0].repeat(2, 1, 1, 1)
    )

    if args.noise_offset:
        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
        preference_noise += args.noise_offset * torch.randn(
            (
                preference_model_input.shape[0] // 2,
                preference_model_input.shape[1],
                1,
                1,
            ),
            device=preference_model_input.device,
        )

    # Add noise to the model input according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    preference_bsz = preference_model_input.shape[0] // 2
    preference_timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (preference_bsz,),
        device=preference_model_input.device,
        dtype=torch.long,
    ).repeat(2)

    preference_timesteps = preference_timesteps.long()

    noisy_preference_model_input = noise_scheduler.add_noise(
        preference_model_input, preference_noise, preference_timesteps
    )

    preference_add_time_ids = torch.cat(
        [
            compute_time_ids(s, c, accelerator, weight_dtype)
            for s, c in zip(
                batch["original_sizes"],
                batch["crop_top_lefts"],
            )
        ]
    ).repeat(2, 1)

    # Get the text embedding for conditioning
    preference_prompt_embeds, preference_pooled_prompt_embeds = encode_prompt(
        text_encoders=[text_encoder_one, text_encoder_two],
        tokenizers=None,
        prompt=None,
        text_input_ids_list=[
            batch["input_ids_one"],
            batch["input_ids_two"],
        ],
    )
    preference_prompt_embeds = preference_prompt_embeds.repeat(2, 1, 1)
    preference_pooled_prompt_embeds = preference_pooled_prompt_embeds.repeat(2, 1)

    # Predict the noise residual
    preference_model_pred = unet(
        noisy_preference_model_input,
        preference_timesteps,
        preference_prompt_embeds,
        added_cond_kwargs={
            "time_ids": preference_add_time_ids,
            "text_embeds": preference_pooled_prompt_embeds,
        },
    ).sample

    # Get the target for loss depending on the prediction type
    if args.prediction_type is not None:
        # set prediction_type of scheduler if defined
        noise_scheduler.register_to_config(prediction_type=args.prediction_type)

    if noise_scheduler.config.prediction_type == "epsilon":
        preference_target = preference_noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        preference_target = noise_scheduler.get_velocity(
            preference_model_input, preference_noise, preference_timesteps
        )
    else:
        raise ValueError(
            f"Unknown prediction type {noise_scheduler.config.prediction_type}"
        )

    # computing contrastive-dpo loss (P(X_r|C_r) >> P(X_f | C_r))

    # Compute losses.
    model_contrastive_losses = F.mse_loss(
        preference_model_pred.float(),
        preference_target.float(),
        reduction="none",
    )
    model_contrastive_losses = model_contrastive_losses.mean(
        dim=list(range(1, len(model_contrastive_losses.shape)))
    )
    model_contrastive_losses_w, model_contrastive_losses_l = (
        model_contrastive_losses.chunk(2)
    )

    # For logging
    raw_model_contrastive_loss = 0.5 * (
        model_contrastive_losses_w.mean() + model_contrastive_losses_l.mean()
    )
    model_contrastive_diff = (
        model_contrastive_losses_w - model_contrastive_losses_l
    )  # These are both LBS (as is t)
    if not args.disable_ref_model:
        # Reference model predictions.
        if not args.use_ema_ref:
            accelerator.unwrap_model(unet).disable_adapters()
            with torch.no_grad():
                ref_preds = unet(
                    noisy_preference_model_input,
                    preference_timesteps,
                    preference_prompt_embeds,
                    added_cond_kwargs={
                        "time_ids": preference_add_time_ids,
                        "text_embeds": preference_pooled_prompt_embeds,
                    },
                ).sample
                ref_loss = F.mse_loss(
                    ref_preds.float(),
                    preference_target.float(),
                    reduction="none",
                )
                ref_loss = ref_loss.mean(dim=list(range(1, len(ref_loss.shape))))

                ref_losses_w, ref_losses_l = ref_loss.chunk(2)
                ref_diff = ref_losses_w - ref_losses_l
                raw_ref_contrastive_loss = ref_loss.mean()  # for logging
                raw_ref_contrastive_loss = raw_ref_contrastive_loss.detach().item()
        else:
            # Use ema_unet directly as reference model
            with torch.no_grad():
                assert ema_ref_unet is not None, "EMA reference model is NONE"
                ref_preds = ema_ref_unet(
                    noisy_preference_model_input,
                    preference_timesteps,
                    preference_prompt_embeds,
                    added_cond_kwargs={
                        "time_ids": preference_add_time_ids,
                        "text_embeds": preference_pooled_prompt_embeds,
                    },
                ).sample
                ref_loss = F.mse_loss(
                    ref_preds.float(),
                    preference_target.float(),
                    reduction="none",
                )
                ref_loss = ref_loss.mean(dim=list(range(1, len(ref_loss.shape))))

                ref_losses_w, ref_losses_l = ref_loss.chunk(2)
                ref_diff = ref_losses_w - ref_losses_l
                raw_ref_contrastive_loss = ref_loss.mean()  # for logging
                raw_ref_contrastive_loss = raw_ref_contrastive_loss.detach().item()

    else:
        ref_diff = 0
        raw_ref_contrastive_loss = 0

    # Re-enable adapters.
    if not args.disable_ref_model:
        accelerator.unwrap_model(unet).enable_adapters()

    scale_term = -0.5 * float(cfg["dpo"]["beta_dpo"])
    inside_term = scale_term * (model_contrastive_diff - ref_diff)
    contrastive_dpo_loss = -1 * F.logsigmoid(inside_term).mean()

    implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
    implicit_acc += 0.5 * (inside_term == 0).sum().float() / inside_term.size(0)

    losses = {"contrastive_dpo_loss": contrastive_dpo_loss}
    logs = {
        "implicit_acc": implicit_acc.detach().item(),
        "raw_ref_contrastive_loss": raw_ref_contrastive_loss,
        "raw_model_contrastive_loss": raw_model_contrastive_loss.detach().item(),
        "implicit_acc": implicit_acc.detach().item(),
        "contrastive_dpo_loss": contrastive_dpo_loss.detach().item(),
    }
    return losses, logs


def get_experiment_name(args):
    """Create descriptive experiment name from args."""
    # Join multiple triplets with '+'
    triplets_str = "+".join(args.target_triplets)
    name_parts = [triplets_str]

    # Add training mode
    name_parts.append(args.training_mode)

    # Add dataset type
    if args.use_freq_rare:
        name_parts.append("freq_rare")
    else:
        name_parts.append("intermediate")

    # Add seed if used
    if args.seed is not None:
        name_parts.append(f"seed{args.seed}")

    # Join with underscores
    return "_".join(name_parts)


def main(args, cfg):

    cfg["data"]["num_examples_per_intermediate"] = args.num_examples_per_intermediate
    cfg["data"]["intermediate_types"] = args.intermediate_types
    cfg["data"]["num_intermediates_per_relation"] = args.num_intermediates_per_relation
    # cfg['data']['intermediate_types'] = ["active"]
    print("AFTER REPLACING ARGS in CFG fopr data")

    print(cfg)

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )
    target_tiplets_output = "_".join(args.target_triplets)

    if (
        args.weighted_sup
        or cfg["data"]["num_examples_per_intermediate"]
        or cfg["data"]["num_intermediates_per_relation"]
        or cfg["data"]["intermediate_types"]
    ) and not args.use_freq_rare:
        intermeidate_type_str = "_".join(cfg["data"]["intermediate_types"])
        args.output_dir = Path(
            args.output_dir,
            f"{args.training_mode}_weighted_sup_ActiveWeight{args.active_weight}_numExamplePerIntermediate{cfg['data']['num_examples_per_intermediate']}_numIntermediatesPerRelation{cfg['data']['num_intermediates_per_relation']}_intermediateTypes{intermeidate_type_str}",
        )

    else:
        args.output_dir = Path(args.output_dir, args.training_mode)

    args.output_dir = Path(args.output_dir, target_tiplets_output)

    if args.use_freq_rare:
        args.output_dir = Path(args.output_dir, "use_freq_rare")
    else:
        args.output_dir = Path(args.output_dir, "intermediate")

    # Clean up specific experiment directory if starting fresh
    if not args.eval_only and args.resume_from_checkpoint is None:
        if os.path.exists(args.output_dir):
            print(
                f"Starting fresh training run - removing existing experiment directory: {args.output_dir}"
            )
            shutil.rmtree(args.output_dir)
            print(f"Removed old experiment directory")

        # Create fresh experiment directory
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Created fresh experiment directory at {args.output_dir}")

    logging_dir = Path(args.output_dir, args.logging_dir)

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id

    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
        variant=args.variant,
    )
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        variant=args.variant,
    )

    ema_ref_unet = UNet2DConditionModel(**unet.config)
    ema_ref_unet.load_state_dict(unet.state_dict())
    ema_ref_unet.train()

    # We only train the additional adapter LoRA layers
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)
    ema_ref_unet.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=torch.float32)
    ema_ref_unet.to(accelerator.device, dtype=weight_dtype)
    # if args.pretrained_vae_model_name_or_path is None:
    #     vae.to(accelerator.device, dtype=torch.float32)
    # else:
    #     vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    if args.enable_npu_flash_attention:
        if is_torch_npu_available():
            logger.info("npu flash attention enabled.")
            unet.enable_npu_flash_attention()
        else:
            raise ValueError(
                "npu flash attention requires torch_npu extensions and is supported only on npu devices."
            )

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    # now we will add new LoRA weights to the attention layers
    # Set correct lora layers
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    ref_unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(unet_lora_config)
    ema_ref_unet.add_adapter(ref_unet_lora_config)
    ema_ref_unet.load_state_dict(unet.state_dict())
    ema_ref_unet.train()
    ema_ref_unet.requires_grad_(False)

    # The text encoder comes from 🤗 transformers, we will also attach adapters to it.
    if args.train_text_encoder:
        # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
        text_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder_one.add_adapter(text_lora_config)
        text_encoder_two.add_adapter(text_lora_config)

    def unwrap_model(model):
        if is_compiled_module(model):
            print("original model is used for validaiton!!")
        else:
            print("lora model is loaded fro validations")
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder attn layers
            unet_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None
            text_encoder_two_lora_layers_to_save = None

            for model in models:
                if isinstance(unwrap_model(model), type(unwrap_model(unet))):
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model)
                    )
                elif isinstance(
                    unwrap_model(model), type(unwrap_model(text_encoder_one))
                ):
                    text_encoder_one_lora_layers_to_save = (
                        convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(model)
                        )
                    )
                elif isinstance(
                    unwrap_model(model), type(unwrap_model(text_encoder_two))
                ):
                    text_encoder_two_lora_layers_to_save = (
                        convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(model)
                        )
                    )
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

            StableDiffusionXLPipeline.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_one_ = None
        text_encoder_two_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(unet))):
                unet_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_one))):
                text_encoder_one_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_two))):
                text_encoder_two_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, _ = StableDiffusionLoraLoaderMixin.lora_state_dict(input_dir)
        unet_state_dict = {
            f'{k.replace("unet.", "")}': v
            for k, v in lora_state_dict.items()
            if k.startswith("unet.")
        }
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(
            unet_, unet_state_dict, adapter_name="default"
        )
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        if args.train_text_encoder:
            _set_state_dict_into_text_encoder(
                lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_one_
            )

            _set_state_dict_into_text_encoder(
                lora_state_dict,
                prefix="text_encoder_2.",
                text_encoder=text_encoder_two_,
            )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [unet_]
            if args.train_text_encoder:
                models.extend([text_encoder_one_, text_encoder_two_])
            cast_training_params(models, dtype=torch.float32)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()
            text_encoder_two.gradient_checkpointing_enable()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [unet]
        if args.train_text_encoder:
            models.extend([text_encoder_one, text_encoder_two])
        cast_training_params(models, dtype=torch.float32)

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
    if args.train_text_encoder:
        params_to_optimize = (
            params_to_optimize
            + list(filter(lambda p: p.requires_grad, text_encoder_one.parameters()))
            + list(filter(lambda p: p.requires_grad, text_encoder_two.parameters()))
        )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    #  TODO (CHECK): removed by sina
    # if args.dataset_name is not None:
    #     # Downloading and loading a dataset from the hub.
    #     dataset = load_dataset(
    #         args.dataset_name,
    #         args.dataset_config_name,
    #         cache_dir=args.cache_dir,
    #         data_dir=args.train_data_dir,
    #     )
    # else:
    #     data_files = {}
    #     if args.train_data_dir is not None:
    #         data_files["train"] = os.path.join(args.train_data_dir, "**")
    #     dataset = load_dataset(
    #         "imagefolder",
    #         data_files=data_files,
    #         cache_dir=args.cache_dir,
    #     )
    # See more about loading custom images at
    # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    if args.use_freq_rare:
        raise NotImplementedError("Frequency rare is not implemented for SDXL")
        preference_data, active_dataset, passive_dataset = create_freq_rare_dpo_dataset(
            data_dir=args.data_dir,
            target_triplet=args.target_triplets[0],
            cfg=cfg,
            args=args,
        )
    else:
        preference_data, active_dataset, passive_dataset = (
            create_intermediate_dpo_dataset(
                data_dir=args.data_dir, target_triplet=args.target_triplets[0], cfg=cfg
            )
        )

    # Check if either dataset is empty and handle it
    if len(active_dataset) == 0 and "active" in cfg["data"]["intermediate_types"]:
        logger.warning(
            "Active dataset is empty! Check your data or intermediate_types configuration."
        )

    if len(passive_dataset) == 0 and "passive" in cfg["data"]["intermediate_types"]:
        logger.warning(
            "Passive dataset is empty! Check your data or intermediate_types configuration."
        )

    if len(active_dataset) == 0 and "active" in cfg["data"]["intermediate_types"]:
        raise ValueError(
            "Active dataset is empty! Check your data or intermediate_types configuration."
        )

    if len(passive_dataset) == 0 and "passive" in cfg["data"]["intermediate_types"]:
        raise ValueError(
            "Passive dataset is empty! Check your data or intermediate_types configuration."
        )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        caption_column = args.caption_column
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        tokens_one = tokenize_prompt(tokenizer_one, captions)
        tokens_two = tokenize_prompt(tokenizer_two, captions)
        return tokens_one, tokens_two

    # Preprocessing the datasets.
    train_resize = transforms.Resize(
        args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
    )
    train_crop = (
        transforms.CenterCrop(args.resolution)
        if args.center_crop
        else transforms.RandomCrop(args.resolution)
    )
    train_flip = transforms.RandomHorizontalFlip(p=1.0)
    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        image_column = "image"
        images = [image.convert("RGB") for image in examples[image_column]]
        triplets = examples["triplet"]
        # image aug
        original_sizes = []
        all_images = []
        crop_top_lefts = []
        for image in images:
            original_sizes.append((image.height, image.width))
            image = train_resize(image)
            if args.random_flip and random.random() < 0.5:
                # flip
                image = train_flip(image)
            if args.center_crop:
                y1 = max(0, int(round((image.height - args.resolution) / 2.0)))
                x1 = max(0, int(round((image.width - args.resolution) / 2.0)))
                image = train_crop(image)
            else:
                y1, x1, h, w = train_crop.get_params(
                    image, (args.resolution, args.resolution)
                )
                image = crop(image, y1, x1, h, w)
            crop_top_left = (y1, x1)
            crop_top_lefts.append(crop_top_left)
            image = train_transforms(image)
            all_images.append(image)

        examples["original_sizes"] = original_sizes
        examples["crop_top_lefts"] = crop_top_lefts
        examples["pixel_values"] = all_images
        tokens_one, tokens_two = tokenize_captions(examples)
        examples["input_ids_one"] = tokens_one
        examples["input_ids_two"] = tokens_two
        examples["triplet"] = triplets
        if args.debug_loss:
            fnames = [
                os.path.basename(image.filename)
                for image in examples[image_column]
                if image.filename
            ]
            if fnames:
                examples["filenames"] = fnames
        return examples

    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.5], [0.5])

    def preprocess_train_preference(examples):
        all_pixel_values = []
        images = [image.convert("RGB") for image in examples["image_winner"]]
        original_sizes = [(image.height, image.width) for image in images]
        crop_top_lefts = []

        for image_column in ["image_winner", "image_loser"]:
            images = [image.convert("RGB") for image in examples[image_column]]
            if image_column == "image_loser":
                # Need to bring down the image to the same resolution.
                # This seems like the simplest reasonable approach.
                # "::-1" because PIL resize takes (width, height).
                images = [
                    image.resize(original_sizes[i][::-1])
                    for i, image in enumerate(images)
                ]
            pixel_values = [to_tensor(image) for image in images]
            all_pixel_values.append(pixel_values)

        # Double on channel dim, jpg_y then jpg_w
        im_tup_iterator = zip(*all_pixel_values)
        combined_pixel_values = []
        for im_tup, label_0 in zip(im_tup_iterator, examples["label"]):
            if label_0 == 0:
                logger.info("Warning it looks like the data is flipped?")
                im_tup = im_tup[::-1]

            combined_im = torch.cat(im_tup, dim=0)  # no batch dim

            # Resize.
            combined_im = train_resize(combined_im)

            # Flipping.
            if args.random_flip and random.random() < 0.5:
                combined_im = train_flip(combined_im)

            # Cropping.
            if args.center_crop:
                y1 = max(0, int(round((combined_im.shape[1] - args.resolution) / 2.0)))
                x1 = max(0, int(round((combined_im.shape[2] - args.resolution) / 2.0)))
                combined_im = train_crop(combined_im)
            else:
                y1, x1, h, w = train_crop.get_params(
                    combined_im, (args.resolution, args.resolution)
                )
                combined_im = crop(combined_im, y1, x1, h, w)

            crop_top_left = (y1, x1)
            crop_top_lefts.append(crop_top_left)
            combined_im = normalize(combined_im)
            combined_pixel_values.append(combined_im)

        examples["pixel_values"] = combined_pixel_values
        examples["original_sizes"] = original_sizes
        examples["crop_top_lefts"] = crop_top_lefts
        tokens_one, tokens_two = tokenize_captions(examples)
        examples["input_ids_one"] = tokens_one
        examples["input_ids_two"] = tokens_two
        examples["triplet_winner"] = examples["triplet"]
        examples["triplet_loser"] = examples["triplet"]
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            active_dataset = active_dataset.shuffle(seed=args.seed).select(
                range(args.max_train_samples)
            )
            passive_dataset = passive_dataset.shuffle(seed=args.seed).select(
                range(args.max_train_samples)
            )

        # Set the training transforms
        # if "active" in cfg['data']['intermediate_types']:
        active_dataset = active_dataset.with_transform(preprocess_train)

        passive_dataset = passive_dataset.with_transform(preprocess_train)
        preference_data = preference_data.with_transform(preprocess_train_preference)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids_one = torch.stack([example["input_ids_one"] for example in examples])
        input_ids_two = torch.stack([example["input_ids_two"] for example in examples])

        # Get the original sizes and crop coordinates
        original_sizes = [example["original_sizes"] for example in examples]
        crop_top_lefts = [example["crop_top_lefts"] for example in examples]

        triplets = [example["triplet"] for example in examples]

        return {
            "pixel_values": pixel_values,
            "input_ids_one": input_ids_one,
            "input_ids_two": input_ids_two,
            "original_sizes": original_sizes,
            "crop_top_lefts": crop_top_lefts,
            "triplet": triplets,
        }

    def collate_fn_preference(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        original_sizes = [example["original_sizes"] for example in examples]
        crop_top_lefts = [example["crop_top_lefts"] for example in examples]
        input_ids_one = torch.stack([example["input_ids_one"] for example in examples])
        input_ids_two = torch.stack([example["input_ids_two"] for example in examples])
        triplets = [example["triplet_winner"] for example in examples]  # todo

        return {
            "pixel_values": pixel_values,
            "input_ids_one": input_ids_one,
            "input_ids_two": input_ids_two,
            "original_sizes": original_sizes,
            "crop_top_lefts": crop_top_lefts,
            "triplet_winner": triplets,
        }

    # DataLoaders creation:
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     shuffle=True,
    #     collate_fn=collate_fn,
    #     batch_size=args.train_batch_size,
    #     num_workers=args.dataloader_num_workers,
    # )

    active_dataloader = torch.utils.data.DataLoader(
        active_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    passive_dataloader = torch.utils.data.DataLoader(
        passive_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    preference_dataloader = torch.utils.data.DataLoader(
        preference_data,
        shuffle=True,
        collate_fn=collate_fn_preference,
        batch_size=args.train_batch_size * 2,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False

    num_update_steps_per_epoch = math.ceil(
        min(
            len(active_dataloader), len(passive_dataloader)
        )  # , len(preference_dataloader)
        / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        (
            unet,
            text_encoder_one,
            text_encoder_two,
            optimizer,
            active_dataloader,
            passive_dataloader,
            preference_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            unet,
            text_encoder_one,
            text_encoder_two,
            optimizer,
            active_dataloader,
            passive_dataloader,
            preference_dataloader,
            lr_scheduler,
        )
    else:
        (
            unet,
            optimizer,
            active_dataloader,
            passive_dataloader,
            preference_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            unet,
            optimizer,
            active_dataloader,
            passive_dataloader,
            preference_dataloader,
            lr_scheduler,
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        min(len(active_dataloader), len(passive_dataloader), len(preference_dataloader))
        / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        # accelerator.init_trackers("text2image-fine-tune", config=vars(args))
        accelerator.init_trackers(
            "text2image-fine-tune",
            config=vars(args),
            init_kwargs={
                "wandb": {
                    "name": get_experiment_name(
                        args
                    )  # Set the run name in wandb to your chosen run_id
                }
            },
        )
    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples  passive = {len(passive_dataset)}")
    logger.info(f"  Num examples  active = {len(active_dataset)}")
    logger.info(f"  Num examples  preference data = {len(preference_dataloader)}")

    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"update steps per epoch = {num_update_steps_per_epoch}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0
    final_evaluation_steps = []
    print("EVALUATION_STEPS  ", EVALUATION_STEPS)
    for st in EVALUATION_STEPS:
        tmp = math.ceil(st / num_update_steps_per_epoch)
        final_evaluation_steps.append(tmp * num_update_steps_per_epoch)

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
            print(f"Resuming from checkpoint {path}")
        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            # accelerator.load_state(path)

            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # pipeline = StableDiffusionXLPipeline.from_pretrained(
    #     args.pretrained_model_name_or_path,
    # )

    # log_validation(pipeline, args, accelerator, -1)

    # del pipeline
    # torch.cuda.empty_cache()
    print("first_epoch", first_epoch)
    if accelerator.is_main_process:
        if args.eval_only:
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                vae=vae,
                text_encoder=unwrap_model(text_encoder_one),
                text_encoder_2=unwrap_model(text_encoder_two),
                unet=unwrap_model(unet),
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
            )
            num_validation_images = 20
            validation_triplets = [
                triplet
                for target_triplet in args.target_triplets
                for triplet in TEST_TRIPLETS[target_triplet]
            ]
            run_evaluation = True
            if args.resume_from_checkpoint == "latest":
                is_final_validation = True
            else:
                is_final_validation = False
            log_validation(
                pipeline,
                args,
                accelerator,
                args.num_train_epochs,
                step=global_step,
                num_validation_images=num_validation_images,
                validation_triplets=validation_triplets,
                run_evaluation=run_evaluation,
                is_final_validation=is_final_validation,
            )

            evaluator = get_evaluator_class(cfg["eval"]["evaluator_model"])

            evaluate_images(
                evaluator,
                args,
                cfg,
                accelerator,
                args.num_train_epochs,
                step=global_step,
                validation_triplets=validation_triplets,
                is_final_validation=is_final_validation,
            )
            del evaluator
            torch.cuda.empty_cache()

            return
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder_one.train()
            text_encoder_two.train()
        train_loss = 0.0
        train_loss_sup = 0.0
        train_loss_contrastive = 0.0
        for step, (active_batch, passive_batch, preference_batch) in enumerate(
            zip(active_dataloader, passive_dataloader, preference_dataloader)
        ):
            # Determine weights based on step

            active_weight = 1.0
            passive_weight = 1.0
            if args.weighted_sup:
                active_weight = exp_ramp_up(
                    0.5, args.active_weight, args.max_train_steps, global_step
                )
            # Create weight tensor matching the batch size
            batch_weights = torch.cat(
                [
                    torch.full(
                        (len(active_batch["pixel_values"]),),
                        active_weight,
                        device=accelerator.device,
                    ),
                    torch.full(
                        (len(passive_batch["pixel_values"]),),
                        passive_weight,
                        device=accelerator.device,
                    ),
                ]
            )

            batch_active_passive = {
                "pixel_values": torch.cat(
                    [active_batch["pixel_values"], passive_batch["pixel_values"]]
                ),
                "input_ids_one": torch.cat(
                    [active_batch["input_ids_one"], passive_batch["input_ids_one"]]
                ),
                "input_ids_two": torch.cat(
                    [active_batch["input_ids_two"], passive_batch["input_ids_two"]]
                ),
                "original_sizes": active_batch["original_sizes"]
                + passive_batch["original_sizes"],
                "crop_top_lefts": active_batch["crop_top_lefts"]
                + passive_batch["crop_top_lefts"],
                "weights": batch_weights,  # Add weights to batch
            }

            # if "filenames" in active_batch and "filenames" in passive_batch:
            #     batch["filenames"] = active_batch["filenames"] + passive_batch["filenames"]

            with accelerator.accumulate(unet):
                if "sup" in args.training_mode:
                    losses_active_passive, logs_active_passive = (
                        train_step_active_passive(
                            batch_active_passive,
                            vae,
                            unet,
                            noise_scheduler,
                            text_encoder_one,
                            text_encoder_two,
                            tokenizer_one,
                            tokenizer_two,
                            accelerator,
                            global_step,
                            weight_dtype,
                            args,
                            cfg,
                        )
                    )
                else:
                    logs_active_passive = {
                        "active_loss": 0,
                        "passive_loss": 0,
                        "sup_loss": 0,
                        "avg_sup_loss": 0,
                    }
                    losses_active_passive = {"sup_loss": 0}

                # ---- preference training
                if "contrastive_dpo" in args.training_mode:
                    losses_contrastive_dpo, logs_contrastive_dpo = (
                        train_step_contrastive_dpo(
                            preference_batch,
                            vae,
                            unet,
                            (
                                ema_ref_unet
                                if not args.disable_ref_model and args.use_ema_ref
                                else None
                            ),
                            noise_scheduler,
                            text_encoder_one,
                            text_encoder_two,
                            accelerator,
                            global_step,
                            weight_dtype,
                            args,
                            cfg,
                        )
                    )
                else:
                    losses_contrastive_dpo = {"contrastive_dpo_loss": 0}
                    logs_contrastive_dpo = {
                        "implicit_acc": 0,
                        "raw_ref_contrastive_loss": 0,
                        "raw_model_contrastive_loss": 0,
                        "implicit_acc": 0,
                        "contrastive_dpo_loss": 0,
                    }

                # shared part !
                # Final loss.

                train_loss_sup = (
                    logs_active_passive["avg_sup_loss"]
                    / args.gradient_accumulation_steps
                )
                train_loss_contrastive = (
                    logs_contrastive_dpo["contrastive_dpo_loss"]
                    / args.gradient_accumulation_steps
                )
                train_loss = train_loss_sup + train_loss_contrastive

                contrastive_dpo_weight = 0 if global_step < 200 else 0.5
                if "contrastive_dpo" not in args.training_mode:
                    contrastive_dpo_weight = 0
                elif "sup" not in args.training_mode:
                    contrastive_dpo_weight = 1
                else:
                    max_step = args.max_train_steps
                    current_step = global_step
                    contrastive_dpo_weight = exp_ramp_up(0, 0.5, max_step, current_step)

                total_loss = (
                    losses_active_passive["sup_loss"]
                    + contrastive_dpo_weight
                    * losses_contrastive_dpo["contrastive_dpo_loss"]
                )
                # Backpropagate
                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if args.use_ema:
                    update_ema(
                        target_params=None,
                        source_params=unet.parameters(),
                        rate=args.ema_decay,
                    )

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # Update EMA model parameters
                if args.use_ema_ref:
                    update_ema(
                        target_params=ema_ref_unet.parameters(),
                        source_params=unet.parameters(),
                        rate=args.ema_decay,
                    )

                progress_bar.update(1)
                global_step += 1
                train_logs = {
                    "train_loss": train_loss,
                    "train_loss_sup": train_loss_sup,
                    "train_loss_contrastive": train_loss_contrastive,
                    "active_loss": logs_active_passive["active_loss"],
                    "passive_loss": logs_active_passive["passive_loss"],
                }
                accelerator.log(train_logs, step=global_step)

                train_loss = 0.0
                train_loss_sup = 0.0
                train_loss_contrastive = 0.0

                # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
                if (
                    accelerator.distributed_type == DistributedType.DEEPSPEED
                    or accelerator.is_main_process
                ):
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {
                "step loss": total_loss.detach().item(),
                "contrastive loss": logs_contrastive_dpo["contrastive_dpo_loss"],
                "sup loss": logs_active_passive["sup_loss"],
                "raw_model_loss": logs_contrastive_dpo["raw_model_contrastive_loss"],
                "ref_loss": logs_contrastive_dpo["raw_ref_contrastive_loss"],
                "implicit_acc": logs_contrastive_dpo["implicit_acc"],
                "lr": lr_scheduler.get_last_lr()[0],
                "contrastive_loss_weight": contrastive_dpo_weight,
                "active_weight": active_weight,
                "passive_weight": passive_weight,
                "global_step": global_step,
                "active_loss": logs_active_passive["active_loss"],
                "passive_loss": logs_active_passive["passive_loss"],
            }
            progress_bar.set_postfix(**logs)
            # accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            # print("BEFORE VALIDATION, GLOBAL STEP: ", global_step)
            if (
                epoch % args.validation_epochs == 0
                or global_step in final_evaluation_steps
            ):

                # print(f"************* VALIDATION : {global_step} *************")
                # # create pipeline

                # print("**BEFORE**Checking LoRA on text_encoder:")
                # check_lora_attached(unwrap_model(text_encoder_one), "text_encoder")

                # print("**BEFORE**Checking LoRA on text_encoder_2:")
                # check_lora_attached(unwrap_model(text_encoder_two), "text_encoder_2")

                # print("**BEFORE**Checking LoRA on UNet:")
                # check_lora_attached(unwrap_model(unet), "UNet")

                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    vae=vae,
                    text_encoder=unwrap_model(text_encoder_one),
                    text_encoder_2=unwrap_model(text_encoder_two),
                    unet=unwrap_model(unet),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )
                # print("Checking LoRA on text_encoder:")
                # check_lora_attached(pipeline.text_encoder, "text_encoder")

                # print("Checking LoRA on text_encoder_2:")
                # check_lora_attached(pipeline.text_encoder_2, "text_encoder_2")

                # print("Checking LoRA on UNet:")
                # check_lora_attached(pipeline.unet, "UNet")

                run_evaluation = (
                    True if global_step in final_evaluation_steps else False
                )

                if run_evaluation:
                    num_validation_images = 10
                    validation_triplets = [
                        triplet
                        for target_triplet in args.target_triplets
                        for triplet in TEST_TRIPLETS[target_triplet]
                    ]
                else:
                    num_validation_images = args.num_validation_images
                    validation_triplets = [
                        triplet
                        for target_triplet in args.target_triplets
                        for triplet in VALIDATION_TRIPLETS[target_triplet]
                    ]

                log_validation(
                    pipeline,
                    args,
                    accelerator,
                    epoch,
                    step=global_step,
                    num_validation_images=num_validation_images,
                    validation_triplets=validation_triplets,
                    run_evaluation=run_evaluation,
                    is_final_validation=False,
                )

                del pipeline
                torch.cuda.empty_cache()

                if run_evaluation:
                    evaluator = get_evaluator_class(cfg["eval"]["evaluator_model"])
                    evaluate_images(
                        evaluator,
                        args,
                        cfg,
                        accelerator,
                        epoch,
                        global_step,
                        validation_triplets=validation_triplets,
                    )

                    del evaluator
                    torch.cuda.empty_cache()

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unet)
        )

        if args.train_text_encoder:
            text_encoder_one = unwrap_model(text_encoder_one)
            text_encoder_two = unwrap_model(text_encoder_two)

            text_encoder_lora_layers = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(text_encoder_one)
            )
            text_encoder_2_lora_layers = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(text_encoder_two)
            )
        else:
            text_encoder_lora_layers = None
            text_encoder_2_lora_layers = None

        StableDiffusionXLPipeline.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_state_dict,
            text_encoder_lora_layers=text_encoder_lora_layers,
            text_encoder_2_lora_layers=text_encoder_2_lora_layers,
        )

        del unet
        del text_encoder_one
        del text_encoder_two
        del text_encoder_lora_layers
        del text_encoder_2_lora_layers
        torch.cuda.empty_cache()

        # Final inference
        # Make sure vae.dtype is consistent with the unet.dtype
        # if args.mixed_precision == "fp16":
        #     vae.to(weight_dtype)
        # Load previous pipeline
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=vae,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )

        # load attention processors
        pipeline.load_lora_weights(args.output_dir)
        pipeline = pipeline.to(accelerator.device, dtype=torch.float32)

        # run inference
        if args.num_validation_images > 0:
            validation_triplets = [
                triplet
                for target_triplet in args.target_triplets
                for triplet in TEST_TRIPLETS[target_triplet]
            ]

            log_validation(
                pipeline,
                args,
                accelerator,
                epoch=args.num_train_epochs,
                step=args.max_train_steps,
                num_validation_images=20,
                validation_triplets=validation_triplets,
                run_evaluation=True,
                is_final_validation=True,
            )
            del pipeline
            torch.cuda.empty_cache()
            accelerator.wait_for_everyone()
            evaluator = get_evaluator_class(cfg["eval"]["evaluator_model"])
            evaluate_images(
                evaluator,
                args,
                cfg,
                accelerator,
                epoch=args.num_train_epochs,
                step=args.max_train_steps,
                validation_triplets=validation_triplets,
                is_final_validation=True,
            )
            del evaluator
            torch.cuda.empty_cache()

        if args.push_to_hub:
            save_model_card(
                repo_id,
                base_model=args.pretrained_model_name_or_path,
                dataset_name=args.dataset_name,
                train_text_encoder=args.train_text_encoder,
                repo_folder=args.output_dir,
                vae_path=args.pretrained_vae_model_name_or_path,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


def check_lora_attached(model, name=""):
    for name, param in model.named_parameters():
        if "lora" in name:
            print(f"[✔] LoRA Found: {name}")
            return True
    print(f"[❌] No LoRA found in {name}")
    return False


if __name__ == "__main__":
    args = parse_args()

    cfg = load_config("./configs/config_sdxl_finetune_contrastive.ini")
    main(args, cfg)
