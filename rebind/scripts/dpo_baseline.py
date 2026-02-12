#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 bram-w, The HuggingFace Inc. team. All rights reserved.
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

import argparse
import contextlib
import io
import logging
import random
from os import times
from venv import create

import math
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from utils.utils import load_init_image_score_pairs_initPrompt, load_softprompt_image_score_pairs, save_init_image_score_pairs_initPrompt, save_softprompt_image_score_pairs
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset, Dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import re

from dataset.DynamicDataset import DynamicDataset

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers
from diffusers.utils.import_utils import is_xformers_available


# sina add:
# from modeling.utils.scene_graph_generator import (
#     sg_check_VQA_train,
#     sg_check_VQA_train_batch,,
# )
from itertools import product
import warnings
from accelerate.utils import broadcast

import pickle

from transformers import CLIPTextModel, CLIPTextModelWithProjection
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch.nn as nn
import torch


from eval.eval import (get_mllm_QA_score, get_normalized_relation_score, sg_check_VQA_train_prompt, get_relation_entropy_scores)
from modeling.utils.internVL_utils import get_model_internVL
from modeling.utils.llava_utils import get_model as get_model_llava


warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.25.0.dev0")

logger = get_logger(__name__)

VALIDATION_PROMPTS = [
    "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
    "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
]

VALIDATION_PROMPTS = [
    "A photo of a mouse chasing  a cat",
    "A photo of a cat chasing a mouse",
]

VALIDATION_PROMPTS_SCENE_GRAPHS = [
    [("mouse", "chasing", "cat")],
    [("cat", "chasing", "mouse")],
]


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def log_validation(
    args, unet, accelerator, weight_dtype, epoch, is_final_validation=False
):
    logger.info(
        f"Running validation... \n Generating images with prompts:\n"
        f" {VALIDATION_PROMPTS}."
    )

    # create pipeline
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    if not is_final_validation:
        pipeline.unet = accelerator.unwrap_model(unet)
    else:
        pipeline.load_lora_weights(
            args.output_dir, weight_name="pytorch_lora_weights.safetensors"
        )

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = (
        torch.Generator(device=accelerator.device).manual_seed(args.seed)
        if args.seed
        else None
    )
    images = []
    context = (
        contextlib.nullcontext() if is_final_validation else torch.cuda.amp.autocast()
    )
    # processor_llava, model_llava = get_model("llava-hf/llava-v1.6-vicuna-7b-hf", quantization=args.llava_quantization)
    # scores = []

    for idx in range(len(VALIDATION_PROMPTS)):
        with context:
            prompt = VALIDATION_PROMPTS[idx]
            scene_graph = VALIDATION_PROMPTS_SCENE_GRAPHS[idx]
            image = pipeline(
                prompt, num_inference_steps=25, generator=generator
            ).images[0]
            # score = llava_score(image,scene_graph, model_llava, processor_llava, accelerator, quantization=args.llava_quantization)
            # scores.append(score)
            images.append(image)

    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(tracker_key, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    tracker_key: [
                        wandb.Image(image, caption=f"{i}: {VALIDATION_PROMPTS[i]}")
                        for i, image in enumerate(images)
                    ]
                }
            )

    # Also log images without the LoRA params for comparison.
    if is_final_validation:
        pipeline.disable_lora()
        no_lora_images = [
            pipeline(prompt, num_inference_steps=25, generator=generator).images[0]
            for prompt in VALIDATION_PROMPTS
        ]

        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img) for img in no_lora_images])
                tracker.writer.add_images(
                    "test_without_lora", np_images, epoch, dataformats="NHWC"
                )
            if tracker.name == "wandb":
                tracker.log(
                    {
                        "test_without_lora": [
                            wandb.Image(image, caption=f"{i}: {VALIDATION_PROMPTS[i]}")
                            for i, image in enumerate(no_lora_images)
                        ]
                    }
                )



@torch.no_grad()
def log_final_validation(
    args,
    accelerator,
    weight_dtype,
    num_images_per_prompt=20,
):
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )

    pipeline.load_lora_weights(
            args.output_dir, weight_name="pytorch_lora_weights.safetensors"
        )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = (
        torch.Generator(device=accelerator.device).manual_seed(args.seed)
        if args.seed
        else None
    )

    context = torch.cuda.amp.autocast()
    
    log_dir = os.path.join(args.output_dir, "final_results")
    os.makedirs(log_dir, exist_ok=True)
    log_image_dir = os.path.join(log_dir, f"images")
    os.makedirs(log_image_dir, exist_ok=True)
    TEST_PROMPT = "A photo of a mouse chasing a cat"
    images = []
    for image_num in range(num_images_per_prompt):
        with context:
            image = pipeline(
                TEST_PROMPT, num_inference_steps=25, generator=generator
            ).images[0]
            images.append(image)
            image.save(os.path.join(log_image_dir, f"{image_num}.png"))
            

            

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
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
        "--dataset_split_name",
        type=str,
        default="validation",
        help="Dataset split to be used during training. Helpful to specify for conducting experimental runs.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--run_validation",
        default=False,
        action="store_true",
        help="Whether to run validation inference in between training and also after training. Helps to track progress.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=200,
        help="Run validation every X steps.",
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
        default="diffusion-dpo-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--vae_encode_batch_size",
        type=int,
        default=8,
        help="Batch size to use for VAE encoding of the images for efficient processing.",
    )
    parser.add_argument(
        "--no_hflip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--random_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to random crop the input images to the resolution. If not set, the images will be center-cropped."
        ),
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
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
        "--beta_dpo",
        type=int,
        default=2500,
        help="DPO KL Divergence penalty.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="sigmoid",
        help="DPO loss type. Can be one of 'sigmoid' (default), 'ipo', or 'cpo'",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
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
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
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
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
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
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
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
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--tracker_name",
        type=str,
        default="diffusion-dpo-lora",
        help=("The name of the tracker to report results to."),
    )
    # added by sina
    parser.add_argument("--dataset_generator_step", type=int, default=None)

    parser.add_argument("--combine_dpoLoss_supervisedLoss", action="store_true")

    parser.add_argument("--llava_quantization", action="store_true")


    parser.add_argument("--MLLM_evaluator", type=str, default="llava")
    parser.add_argument("--init_prompt", type=str, default=None)
    parser.add_argument("--eval_goal", type=str, default=None)
    parser.add_argument(
        "--path_init_image_score_pairs",
        type=str,
        default="/ix/akovashka/sem238/image_score_pairs/init_prompts",
    )
    parser.add_argument("--VLM_n_questions", type=int, default=9)
    parser.add_argument("--eval_only", action="store_true")

    
    # rgs.dynamically_update_dataset and epoch % args.dataset_generator_step

    # finished added by sina

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None:
        raise ValueError("Must provide a `dataset_name`.")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def tokenize_captions(tokenizer, examples):
    max_length = tokenizer.model_max_length
    captions = []
    for caption in examples["caption"]:
        captions.append(caption)

    text_inputs = tokenizer(
        captions,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    # print("text inputs ", text_inputs)
    return text_inputs.input_ids


@torch.no_grad()
def encode_prompt(text_encoder, input_ids):
    text_input_ids = input_ids.to(text_encoder.device)
    attention_mask = None
    # print("ids:  ", text_input_ids)
    # print("attention_mask  , ", attention_mask.shape)
    prompt_embeds = text_encoder(text_input_ids, attention_mask=attention_mask)
    print("embeds - " * 5 + "\n", prompt_embeds)
    print("prompt_embeds.last_hidden_state ", prompt_embeds.last_hidden_state.shape)
    print("prompt_embeds.pooler_output ", prompt_embeds.pooler_output.shape)
    prompt_embeds = prompt_embeds[0]
    print("prompt_embeds[0]  ", prompt_embeds.shape)

    return prompt_embeds


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    experiment_name = f"INITPROMPT{re.sub(r'\s+', '_', args.init_prompt)}_EVALGOAL{args.eval_goal}_MLLM{args.MLLM_evaluator}NUM_QUESTIONS{args.VLM_n_questions}"
    args.output_dir = os.path.join(args.output_dir, experiment_name)

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    num_devices = accelerator.num_processes
    print(f"Number of available devices: {num_devices}")

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            if os.path.exists(args.output_dir) and not args.eval_only:
                # Remove the existing directory and its contents
                shutil.rmtree(args.output_dir)
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
    )
    print("MODEl", type(text_encoder))
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        variant=args.variant,
    )

    # making models requries fgrad false
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Set up LoRA.
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    # Add adapter and make sure the trainable params are in float32.
    unet.add_adapter(unet_lora_config)
    if args.mixed_precision == "fp16":
        for param in unet.parameters():
            # only upcast trainable parameters (LoRA) into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)

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

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder atten layers
            unet_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(accelerator.unwrap_model(unet))):
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(model)
                    )
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            StableDiffusionLoraLoaderMixin.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=None,
            )

    def load_model_hook(models, input_dir):
        unet_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(accelerator.unwrap_model(unet))):
                unet_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, network_alphas = (
            StableDiffusionLoraLoaderMixin.lora_state_dict(input_dir)
        )
        StableDiffusionLoraLoaderMixin.load_lora_into_unet(
            lora_state_dict, network_alphas=network_alphas, unet=unet_
        )

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

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
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    # train_dataset = load_dataset(
    #     args.dataset_name,
    #     cache_dir=args.cache_dir,
    #     split=args.dataset_split_name,
    # )

    train_transforms = transforms.Compose(
        [
            transforms.Resize(
                int(args.resolution),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            (
                transforms.RandomCrop(args.resolution)
                if args.random_crop
                else transforms.CenterCrop(args.resolution)
            ),
            (
                transforms.Lambda(lambda x: x)
                if args.no_hflip
                else transforms.RandomHorizontalFlip()
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train_original(examples):
        all_pixel_values = []
        for col_name in ["jpg_0", "jpg_1"]:
            images = [
                Image.open(io.BytesIO(im_bytes)).convert("RGB")
                for im_bytes in examples[col_name]
            ]
            pixel_values = [train_transforms(image) for image in images]
            all_pixel_values.append(pixel_values)

        # Double on channel dim, jpg_y then jpg_w
        im_tup_iterator = zip(*all_pixel_values)
        combined_pixel_values = []
        for im_tup, label_0 in zip(im_tup_iterator, examples["label_0"]):
            if label_0 == 0:
                im_tup = im_tup[::-1]
            combined_im = torch.cat(im_tup, dim=0)  # no batch dim
            combined_pixel_values.append(combined_im)
        examples["pixel_values"] = combined_pixel_values

        examples["input_ids"] = tokenize_captions(tokenizer, examples)
        return examples

    def preprocess_train(examples):
        all_pixel_values = []
        for col_name in ["jpg_0", "jpg_1"]:
            # images = [Image.open(io.BytesIO(im_bytes)).convert("RGB") for im_bytes in examples[col_name]]
            images = [im.convert("RGB") for im in examples[col_name]]
            pixel_values = [train_transforms(image) for image in images]
            all_pixel_values.append(pixel_values)

        # Double on channel dim, jpg_y then jpg_w
        im_tup_iterator = zip(*all_pixel_values)
        combined_pixel_values = []
        delta_scores = []
        data = {}
        for im_tup, label_0, id_0, id_1 in zip(
            im_tup_iterator,
            examples["delta_score"],
            examples["idx_0"],
            examples["idx_1"],
        ):
            if label_0 < 0:
                print("AJAB :) ")
                im_tup = im_tup[::-1]
            combined_im = torch.cat(im_tup, dim=0)  # no batch dim
            combined_pixel_values.append(combined_im)
            delta_scores.append(abs(label_0))
        data["pixel_values"] = combined_pixel_values
        data["delta_scores"] = torch.tensor(examples["delta_score"])
        data["delta_scores2"] = examples["delta_score"]
        data["id_w"] = examples["idx_0"]
        data["id_l"] = examples["idx_1"]
        data["input_ids"] = tokenize_captions(tokenizer, examples)
        data["score_w"] = examples["score_0"]
        data["score_l"] = examples["score_1"]
        # data["id_w"] = id_0
        # data["id_l"] = id_1
        return data

    # with accelerator.main_process_first():
    #     if args.max_train_samples is not None:
    #         train_dataset = train_dataset.shuffle(seed=args.seed).select(range(args.max_train_samples))
    #     # Set the training transforms
    #     train_dataset = train_dataset.with_transform(preprocess_train)

    # def preprocess_train_online(examples):
    #     all_pixel_values = []
    #     data = {}
    #     for col_name in ["jpg_0", "jpg_1"]:  #score is score_0 - score_1
    #         images = examples[col_name]
    #         pixel_values = [train_transforms(image) for image in images]
    #
    #         all_pixel_values.append(pixel_values)
    #     im_tup_iterator = zip(*all_pixel_values)
    #     combined_pixel_values = []
    #     delta_scores = []
    #     for im_tup, label_0 in zip(im_tup_iterator, examples['score']):
    #         if label_0 < 0:
    #             im_tup = im_tup[::-1]
    #         combined_im = torch.cat(im_tup, dim=0)
    #         combined_pixel_values.append(combined_im)
    #         delta_scores.append(abs(label_0))
    #     data['pixel_values'] = combined_pixel_values
    #     data["input_ids"] = tokenize_captions(tokenizer, examples)
    #     data['delta_scores'] = torch.tensor(delta_scores)
    #     return data

    def collate_fn(examples):
        # print("examples:  ", examples.keys())
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        final_dict = {"pixel_values": pixel_values}
        # print(examples["delta_scores2"])
        final_dict["delta_scores"] = torch.tensor(
            [example["delta_scores2"] for example in examples]
        )
        final_dict["input_ids"] = torch.stack(
            [example["input_ids"] for example in examples]
        )
        final_dict["id_w"] = torch.tensor([example["id_w"] for example in examples])
        final_dict["id_l"] = torch.tensor([example["id_l"] for example in examples])
        final_dict["score_w"] = torch.tensor(
            [example["score_w"] for example in examples]
        )
        final_dict["score_l"] = torch.tensor(
            [example["score_l"] for example in examples]
        )

        return final_dict

    # def collate_fn_online(examples):
    #     pixel_values = torch.stack(examples['pixel_values'])
    #     pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    #     final_dict = {"pixel_values": pixel_values}
    #     final_dict["input_ids"] = examples["input_ids"]
    #     final_dict["delta_scores"] = examples["delta_scores"]
    #     return final_dict

    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=args.train_batch_size,
    #     shuffle=True,
    #     collate_fn=collate_fn,
    #     num_workers=args.dataloader_num_workers,
    # )

    # ADDED BY SINA:

    prompt_train = args.init_prompt  # "A photo of a mouse chasing a cat"
    prompt_train_scene_graph = (
        args.eval_goal
    )  # "mouse_chasing_cat"  # [("cat", "chasing", "mouse")]


    
    prompt_path = re.sub(r"\s+", "_", prompt_train)
    with accelerator.main_process_first():
        image_score_pairs = load_init_image_score_pairs_initPrompt(
            args.init_prompt,
            args.pretrained_model_name_or_path,
            args.MLLM_evaluator,
            args.VLM_n_questions,
            args.path_init_image_score_pairs,
        )
        if len(image_score_pairs) == 0:
            print("INIT PATH NOT FOUND : ", args.path_init_image_score_pairs)
            image_score_pairs = fw_generate_images(
                prompt=prompt_train,
                prompt_scene_graph=prompt_train_scene_graph,
                lora_unet=unet,
                weight_dtype=weight_dtype,
                accelerator=accelerator,
                initial_iteratrion=True,
                num_images=256,
                num_inference_steps=25,
                global_step=0,
                args=args,
            )
            save_init_image_score_pairs_initPrompt(
                image_score_pairs,
                args.init_prompt,
                args.pretrained_model_name_or_path,
                args.MLLM_evaluator,
                args.VLM_n_questions,
                args.path_init_image_score_pairs,
            )
        dynamic_dataset = DynamicDataset(image_score_pairs)
    train_dataset, train_dataloader = create_dataset(
        dynamic_dataset, accelerator, preprocess_train, collate_fn, args
    )

    print(
        f"at gpu {accelerator.process_index}, length of dataset is {len(dynamic_dataset)}"
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Finished ADDED BY SINA:

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    # if accelerator.is_main_process:
    #     accelerator.init_trackers(args.tracker_name, config=vars(args))

    if accelerator.is_main_process:
        run_id = experiment_name
        accelerator.init_trackers(
            args.tracker_name,
            config=vars(args),
            init_kwargs={
                "wandb": {
                    "name": run_id  # Set the run name in wandb to your chosen run_id
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
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
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

    # training loop
    unet.train()

    for epoch in range(first_epoch, args.num_train_epochs):

        for step, batch in enumerate(train_dataloader):

            with accelerator.accumulate(unet):

                # (batch_size, 2*channels, h, w) -> (2*batch_size, channels, h, w)

                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)

                feed_pixel_values = torch.cat(pixel_values.chunk(2, dim=1))
                # print(batch['score_w'])
                all_scores = torch.cat((batch["score_w"], batch["score_l"]), dim=0)
                perfect_image_index = all_scores == 1

                batch["delta_scores"] = batch["delta_scores"]  # todo
                latents = []
                if feed_pixel_values.device != accelerator.device:
                    feed_pixel_values = feed_pixel_values.to(accelerator.device)
                if vae.device != accelerator.device:
                    vae = vae.to(accelerator.device)

                for i in range(
                    0, feed_pixel_values.shape[0], args.vae_encode_batch_size
                ):
                    latents.append(
                        vae.encode(
                            feed_pixel_values[i : i + args.vae_encode_batch_size]
                        ).latent_dist.sample()
                    )
                latents = torch.cat(latents, dim=0)
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents).chunk(2)[0].repeat(2, 1, 1, 1)

                # Sample a random timestep for each image
                bsz = latents.shape[0] // 2
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                    dtype=torch.long,
                ).repeat(2)

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                print("encode_prompt input shape", batch["input_ids"].shape)

                encoder_hidden_states = encode_prompt(
                    text_encoder, batch["input_ids"]
                ).repeat(2, 1, 1)
                print("encode_prompt output shape", encoder_hidden_states.shape)
                # print("model  ", encoder_hidden_states)
                # if unet.device != accelerator.device:
                #     unet = unet.to(accelerator.device)
                # if noisy_model_input.device != accelerator.device:
                #     noisy_model_input = noisy_model_input.to(accelerator.device)
                #
                # if encoder_hidden_states.device != accelerator.device:
                #     encoder_hidden_states = encoder_hidden_states.to(accelerator.device)
                #
                # if timesteps.device != accelerator.device:
                #     print(accelerator.device)
                #     print(timesteps.device)

                # Predict the noise residual
                model_pred = unet(
                    noisy_model_input,
                    timesteps,
                    encoder_hidden_states,
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                # Compute losses.
                model_losses = F.mse_loss(
                    model_pred.float(), target.float(), reduction="none"
                )
                model_losses = model_losses.mean(
                    dim=list(range(1, len(model_losses.shape)))
                )
                if sum(perfect_image_index) >= 4:
                    print("doing supervised training :) ")
                    perfect_losses = model_losses[perfect_image_index]
                    supervised_loss = perfect_losses.mean()
                else:
                    supervised_loss = 0

                model_losses_w, model_losses_l = model_losses.chunk(2)

                # For logging
                raw_model_loss = 0.5 * (model_losses_w.mean() + model_losses_l.mean())
                model_diff = (
                    model_losses_w - model_losses_l
                )  # These are both LBS (as is t)

                # Reference model predictions.
                accelerator.unwrap_model(unet).disable_adapters()

                with torch.no_grad():
                    ref_preds = unet(
                        noisy_model_input,
                        timesteps,
                        encoder_hidden_states,
                    ).sample.detach()
                    ref_loss = F.mse_loss(
                        ref_preds.float(), target.float(), reduction="none"
                    )
                    ref_loss = ref_loss.mean(dim=list(range(1, len(ref_loss.shape))))

                    ref_losses_w, ref_losses_l = ref_loss.chunk(2)
                    ref_diff = ref_losses_w - ref_losses_l
                    raw_ref_loss = ref_loss.mean()

                # Re-enable adapters.
                accelerator.unwrap_model(unet).enable_adapters()

                # Final loss.
                logits = ref_diff - model_diff
                if args.loss_type == "sigmoid":
                    loss = -1 * F.logsigmoid(args.beta_dpo * logits).mean()
                elif args.loss_type == "hinge":
                    loss = torch.relu(1 - args.beta_dpo * logits).mean()
                elif args.loss_type == "ipo":
                    losses = (logits - 1 / (2 * args.beta)) ** 2
                    loss = losses.mean()
                else:
                    raise ValueError(f"Unknown loss type {args.loss_type}")

                implicit_acc = (logits > 0).sum().float() / logits.size(0)
                implicit_acc += 0.5 * (logits == 0).sum().float() / logits.size(0)
                if args.combine_dpoLoss_supervisedLoss:
                    final_loss = loss + supervised_loss
                else:
                    final_loss = loss
                accelerator.backward(final_loss)
                # if accelerator.is_main_process:
                #     for name, param in unet.named_parameters():
                #         if "lora" in name:  # Assuming LoRA parameters have 'lora' in their name
                #             if param.grad is not None:
                #                 print(f"Gradient for {name}: {param.grad.norm().item()}")
                #             else:
                #                 print(f"No gradient for {name}")
                if accelerator.sync_gradients:
                    # print(params_to_optimize)
                    # print(args.max_grad_norm)
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
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

                    if args.run_validation and global_step % args.validation_steps == 0:
                        log_validation(
                            args,
                            unet=unet,
                            accelerator=accelerator,
                            weight_dtype=weight_dtype,
                            epoch=epoch,
                        )

            logs = {
                "loss": loss.detach().item(),
                "final_loss": final_loss.detach().item(),
                "supervised_loss": supervised_loss,
                "raw_model_loss": raw_model_loss.detach().item(),
                "ref_loss": raw_ref_loss.detach().item(),
                "implicit_acc": implicit_acc.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

            # Save the lora layers

        if (
            args.dataset_generator_step is not None
            and epoch % args.dataset_generator_step == 0
        ):
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                new_pairs = load_softprompt_image_score_pairs(args.output_dir, epoch)
                if len(new_pairs) == 0:
                    new_pairs = fw_generate_images(
                        prompt=prompt_train,
                        prompt_scene_graph=prompt_train_scene_graph,
                        lora_unet=unet,
                        weight_dtype=weight_dtype,
                        accelerator=accelerator,
                        initial_iteratrion=False,
                        num_images=16,
                        num_inference_steps=25,
                        global_step=global_step,
                        args=args,
                    )

                    save_softprompt_image_score_pairs(new_pairs, args.output_dir, epoch)
            with accelerator.main_process_first():
                new_pairs = load_softprompt_image_score_pairs(args.output_dir, epoch)
                assert len(new_pairs) > 0, "no new pairs found"
                for pair in new_pairs:
                    dynamic_dataset.add_image_score(pair[0], pair[1])

            train_dataset, train_dataloader = create_dataset(
                dynamic_dataset, accelerator, preprocess_train, collate_fn, args
            )

            print(
                f"at gpu {accelerator.process_index}, length of dataset is {len(dynamic_dataset)}"
            )
            train_dataloader = accelerator.prepare(train_dataloader)


            #
            # if accelerator.is_main_process:
            #     if epoch % 100 == 50:
            #         pairs[0][0].save(f"/ix/akovashka/sem238/test_{epoch}.png")
            # examples = create_pair_data_single_prompt(pairs, k=k, prompt="A photo of a mouse chasing a cat.")
            # # with accelerator.split_between_processes(pairs, apply_padding=True) as split_pairs:
            # #     examples = create_pair_data_single_prompt(split_pairs, k=k,
            # #                                               prompt="A photo of a mouse chasing a cat.")
            #
            # # accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet = unet.to(torch.float32)
        unet_lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unet)
        )

        StableDiffusionLoraLoaderMixin.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_state_dict,
            text_encoder_lora_layers=None,
        )

        # Final validation?
        if args.run_validation:
            log_validation(
                args,
                unet=None,
                accelerator=accelerator,
                weight_dtype=weight_dtype,
                epoch=epoch,
                is_final_validation=True,
            )

            log_final_validation(
                args,
                accelerator=accelerator,
                weight_dtype=weight_dtype,
                num_images_per_prompt=20,
            )


    

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )
    accelerator.end_training()


# @torch.no_grad()
# def llava_score_batch_test(images, model_llava, processor_llava, accelerator, quantization):
#     print("insdie batch score :  ", accelerator.device)
#     # prompt_scene_graph = [("mouse", "chasing", "cat")]
#     prompt_scene_graph = [("cat", "chasing", "mouse")]
#     num_inconsistencies = sg_check_VQA_train_batch(
#         images,
#         prompt_scene_graph,
#         model_llava,
#         processor_llava,
#         accelerator,
#         quantization=quantization,
#     )

#     max_score = 5
#     score = (max_score - num_inconsistencies) / max_score
#     scores = np.maximum(0.001, score)
#     return scores


@torch.no_grad()
def llava_score(
    image, prompt_scene_graph, model_llava, processor_llava, accelerator, quantization
):
    has_inconsistency, inconsistencies = sg_check_VQA_train_prompt(
        image,
        prompt_scene_graph,
        model_llava,
        processor_llava,
        accelerator,
        quantization=quantization,
    )
    num_inconsistency = len(inconsistencies)
    max_score = 5
    assert (
        num_inconsistency <= max_score
    ), f"more than {max_score} inconsistency {num_inconsistency}"
    assert num_inconsistency >= 0, f"negative inconsistency {num_inconsistency}"
    score = (max_score - num_inconsistency) / max_score
    score = max(0.001, score)
    return score


@torch.no_grad()
def fw_generate_images(
    prompt,
    prompt_scene_graph,
    lora_unet,
    weight_dtype,
    accelerator,
    initial_iteratrion,
    num_images,
    num_inference_steps,
    global_step,
    args,
):
    print("prmpt : ", prompt)
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )

    if not initial_iteratrion:
        pipeline.unet = accelerator.unwrap_model(lora_unet)

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    # pipeline.safety_checker = lambda images, clip_input: (images, False)

    # run inference
    generator = (
        torch.Generator(device=accelerator.device).manual_seed(args.seed)
        if args.seed
        else None
    )
    context = torch.cuda.amp.autocast()
    images = []

    for i in range(num_images):
        with context:
            image = pipeline(
                prompt, num_inference_steps=num_inference_steps, generator=generator
            ).images[0]
            images.append(image)
    del pipeline
    torch.cuda.empty_cache()
    if accelerator.is_main_process:
        if args.MLLM_evaluator == "llava":
            processor_eval, model_eval = get_model_llava(
                "llava-hf/llava-v1.6-vicuna-7b-hf", quantization=args.llava_quantization
            )
        elif args.MLLM_evaluator == "internvl":
            processor_eval, model_eval = get_model_internVL(
                "OpenGVLab/InternVL2-8B", quantization=args.llava_quantization
            )
            processor_eval_llava, model_eval_llava = get_model_llava(
                "llava-hf/llava-v1.6-vicuna-7b-hf", quantization=args.llava_quantization
            )
        else:
            raise ValueError(f"model not registered! got: {args.MLLM_evaluator}")

    scores = []
    qa_scores = []
    rel_scores = []
    total_entropy_scores = []
    true_rel_entropy_scores = []
    for image in tqdm(images):
        if accelerator.is_main_process:
            qa_score = get_mllm_QA_score(
                image,
                prompt_scene_graph,
                model_eval,
                processor_eval,
                accelerator.device,
                quantization=args.llava_quantization,
                MLLM_evaluator=args.MLLM_evaluator,
                num_questions=args.VLM_n_questions,
                return_QA_pairs=False,
            )
            rel_score = get_normalized_relation_score(
                image,
                prompt_scene_graph,
                model_eval if args.MLLM_evaluator == "llava" else model_eval_llava,
                (
                    processor_eval
                    if args.MLLM_evaluator == "llava"
                    else processor_eval_llava
                ),
                accelerator.device,
                quantization=args.llava_quantization,
                MLLM_evaluator="llava",
            )
            entropy_score = get_relation_entropy_scores(
                image,
                prompt_scene_graph,
                model_eval if args.MLLM_evaluator == "llava" else model_eval_llava,
                (
                    processor_eval
                    if args.MLLM_evaluator == "llava"
                    else processor_eval_llava
                ),
                accelerator.device,
                quantization=args.llava_quantization,
                MLLM_evaluator="llava",
            )
        score = qa_score  # + rel_score
        scores.append(score)
        qa_scores.append(qa_score)
        rel_scores.append(rel_score)
        total_entropy_scores.append(entropy_score["total_entropy"])
        true_rel_entropy_scores.append(entropy_score["true_relation_entropy"])
    pairs = list(zip(images, scores))
    scores = [pair[1] for pair in pairs]
    if not initial_iteratrion:
        logging.info(
            f"new pairs scores  MEAN {np.mean(scores)}  MIN {np.min(scores)}   MAX {np.max(scores)}"
        )
        logs = {
            "QA_SCORE_MEAN": np.mean(qa_scores),
            "QA_SCORE_MIN": np.min(qa_scores),
            "QA_SCORE_MAX": np.max(qa_scores),
            "REL_SCORE_MEAN": np.mean(rel_scores),
            "REL_SCORE_MIN": np.min(rel_scores),
            "REL_SCORE_MAX": np.max(rel_scores),
            "TOTAL_ENTROPY_SCORE_MEAN": np.mean(total_entropy_scores),
            "TOTAL_ENTROPY_SCORE_MIN": np.min(total_entropy_scores),
            "TOTAL_ENTROPY_SCORE_MAX": np.max(total_entropy_scores),
            "TRUE_REL_ENTROPY_SCORE_MEAN": np.mean(true_rel_entropy_scores),
            "TRUE_REL_ENTROPY_SCORE_MIN": np.min(true_rel_entropy_scores),
            "TRUE_REL_ENTROPY_SCORE_MAX": np.max(true_rel_entropy_scores),
        }

        accelerator.log(logs, step=global_step)

    del processor_eval, model_eval
    torch.cuda.empty_cache()
    return pairs


def create_dataset(dynamic_dataset, accelerator, preprocess_train, collate_fn, args):
    with accelerator.main_process_first():
        data_dict = dynamic_dataset.get_lower_and_upper(args.seed)
        # print(
        #     f"at gpu {accelerator.process_index}, dynamic_dataset {(data_dict['jpg_0'][0], data_dict['jpg_1'][0], data_dict['idx_0'][0], data_dict['idx_1'][0], data_dict['score_0'][0], data_dict['score_1'][0])}")
        train_dataset = Dataset.from_dict(data_dict)

        if args.max_train_samples is not None:
            train_dataset = train_dataset.shuffle(seed=args.seed).select(
                range(args.max_train_samples)
            )
        # Set the training transforms
        train_dataset = train_dataset.with_transform(preprocess_train)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    return train_dataset, train_dataloader


if __name__ == "__main__":
    args = parse_args()
    main(args)
