#!/usr/bin/env python
import argparse
import torch
from pathlib import Path
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--lora_weights_path",
        type=str,
        required=True,
        help="Path to LoRA weights (directory containing 'pytorch_lora_weights.safetensors')",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="demo_outputs",
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=4,
        help="Number of images to generate per prompt",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["a mouse chasing a cat", "a cat chasing a mouse"],
        help="Prompts to generate images for",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load base model
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")

    # Load LoRA weights
    pipeline.load_lora_weights(args.lora_weights_path)
    
    # Generate images for each prompt
    for prompt_idx, prompt in enumerate(args.prompts):
        print(f"\nGenerating {args.num_images} images for prompt: {prompt}")
        
        # Generate one image at a time
        for img_idx in range(args.num_images):
            print(f"Generating image {img_idx + 1}/{args.num_images}")
            
            # Generate single image
            image = pipeline(
                prompt=prompt,
                num_images_per_prompt=1,  # Generate only one image
                num_inference_steps=50,
            ).images[0]  # Get the single generated image

            output_path = os.path.join(
                args.output_dir, 
                prompt.replace(" ", "_"),
            )
            os.makedirs(output_path, exist_ok=True)
            # Save image
            output_path = os.path.join(
                output_path, 
                f"img_{img_idx}.png"
            )
            image.save(output_path)
            print(f"Saved image to {output_path}")
            
            # Clear memory after each image
            torch.cuda.empty_cache()

    # Clean up
    del pipeline
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
