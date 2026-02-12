#!/usr/bin/env python
# coding=utf-8

import argparse
import torch
from diffusers import StableDiffusionXLPipeline
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Demo script for SDXL LoRA inference")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Path to base SDXL model",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to either the directory containing LoRA weights or a specific checkpoint directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt for image generation",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=1,
        help="Number of images to generate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for generation",
    )
    return parser.parse_args()


def find_lora_weights_path(model_path):
    """Find the LoRA weights file in the given directory."""
    # Common LoRA weight filenames
    weight_files = [
        "sd_xl_lora.safetensors",
        "pytorch_lora_weights.safetensors",
        "pytorch_lora_weights.bin",
    ]

    # First check if any of the weight files exist in the given path
    for weight_file in weight_files:
        weights_path = os.path.join(model_path, weight_file)
        if os.path.exists(weights_path):
            return model_path

    # If not found, look for the latest checkpoint
    checkpoints = []
    for dirname in os.listdir(model_path):
        if dirname.startswith("checkpoint-"):
            checkpoint_num = int(dirname.split("-")[1])
            checkpoints.append((checkpoint_num, dirname))

    if not checkpoints:
        raise ValueError(f"No LoRA weights or checkpoints found in {model_path}")

    # Get the latest checkpoint
    latest_checkpoint = sorted(checkpoints, reverse=True)[0][1]
    checkpoint_path = os.path.join(model_path, latest_checkpoint)

    # Verify weights exist in checkpoint directory
    for weight_file in weight_files:
        weights_path = os.path.join(checkpoint_path, weight_file)
        if os.path.exists(weights_path):
            return checkpoint_path

    raise ValueError(f"No LoRA weights found in latest checkpoint: {checkpoint_path}")


def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Find the correct weights path
    try:
        weights_path = find_lora_weights_path(args.model_path)
        print(f"Using LoRA weights from: {weights_path}")
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Load base model
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")

    # Load and fuse LoRA weights
    pipe.load_lora_weights(weights_path)

    # Set up generator for reproducibility if seed is provided
    generator = None
    if args.seed is not None:
        generator = torch.Generator("cuda").manual_seed(args.seed)

    # Generate images
    for i in range(args.num_images):
        image = pipe(
            prompt=args.prompt,
            generator=generator,
            num_inference_steps=50,
        ).images[0]

        # Save the image
        output_path = os.path.join(args.output_dir, f"output_{i}.png")
        image.save(output_path)
        print(f"Saved image to {output_path}")


if __name__ == "__main__":
    main()
