from typing import List
import logging
from PIL import Image
import torch
from diffusers import AuraFlowPipeline
from modeling.generators.generator import T2IGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AuraFlow2(T2IGenerator):
    """Wrapper around AuraFlow implementation"""

    def __init__(
        self,
        model_id: str = "fal/AuraFlow-v0.2",
        device: str = "cuda",
        guidance_scale: float = 3.5,
    ):
        super().__init__()
        self.device = device
        self.guidance_scale = guidance_scale
        logger.info(f"Using device: {self.device}")

        # Load model
        self.pipe = AuraFlowPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            variant="fp16" if self.device == "cuda" else None,
        ).to(self.device)

        # Enable memory optimization
        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()

        # Default negative prompt for better quality
        self.negative_prompt = (
            "blurry, low quality, low resolution, deformed, distorted, disfigured, "
            "watermark, text, logo, duplicate, poorly drawn, bad anatomy, wrong anatomy, "
            "extra limbs, missing limbs, floating limbs, disconnected limbs, mutation, "
            "mutated, ugly, disgusting, bad art, poorly drawn"
        )

    def generate_images(
        self,
        prompts: List[str],
        num_images: int = 1,
        size: str = "1024x1024",
        num_inference_steps: int = 50,
        seed: int = -1,
    ) -> List[Image.Image]:
        """
        Generate images using AuraFlow
        Args:
            prompt: Text prompt
            num_images: Number of images to generate
            size: Image size (e.g., "1024x1024")
            num_inference_steps: Number of denoising steps
            seed: Random seed for generation

        Returns:
            List of generated images
        """
        width, height = map(int, size.split("x"))
        generated_images = []

        try:
            for i in range(num_images):
                # Set up generator for reproducibility
                if seed > 0:
                    generator = torch.Generator(device=self.device).manual_seed(seed)
                else:
                    generator = None

                # Generate image
                output = self.pipe(
                    prompt=prompts[i],
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    # generator=generator,
                    guidance_scale=self.guidance_scale,
                ).images[0]

                # Convert to PIL if needed
                if not isinstance(output, Image.Image):
                    output = Image.fromarray(output)

                generated_images.append(output)

            logger.info(f"Successfully generated {len(generated_images)} images")
            return generated_images

        except Exception as e:
            logger.error(f"Error generating images: {str(e)}")
            return []

    def forward(
        self,
        prompts: List[str],
        num_images: int = 1,
        size: str = "1024x1024",
        seed: int = -1,
    ) -> List[Image.Image]:
        """
        Generate images using AuraFlow (implements T2IGenerator interface)
        """
        return self.generate_images(
            prompts=prompts, num_images=num_images, size=size, seed=seed
        )
