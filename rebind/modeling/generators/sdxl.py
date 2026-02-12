from typing import List
import logging
from PIL import Image
import torch
from diffusers import StableDiffusionXLPipeline
from modeling.generators.generator import T2IGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SDXL(T2IGenerator):
    """Wrapper around Stable Diffusion XL implementation"""

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        logger.info(f"Using device: {self.device}")

        # Load model
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            use_safetensors=True,
            variant="fp16" if self.device == "cuda" else None,
        ).to(self.device)

        # Enable memory optimization
        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_vae_slicing()
        self.negative_prompt = "distorted, deformed, blurry, low quality, low resolution, watermark, text, logo"

    def generate_images(
        self,
        prompts: List[str],
        num_images: int = 1,
        size: str = "512x512",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: int = -1,
    ) -> List[Image.Image]:
        """
        Generate images using SDXL

        Args:
            prompt: Text prompt
            num_images: Number of images to generate
            size: Image size (e.g., "512x512")
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt

        Returns:
            List of generated images
        """
        try:
            # Parse size
            width, height = map(int, size.split("x"))
            generator = torch.Generator(device=self.device)
            if seed > 0:
                generator.manual_seed(seed)

            generated_images = []
            for i in range(0, num_images):
                # Generate images
                output = self.pipe(
                    prompt=prompts[i],
                    negative_prompt=self.negative_prompt,
                    num_images_per_prompt=1,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    generator=generator,
                )

                # Convert to PIL images
                generated_image = output.images[0]
                generated_images.append(generated_image)

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
        Generate images using SDXL (implements T2IGenerator interface)
        """
        return self.generate_images(
            prompts=prompts,
            num_images=num_images,
            size=size,
            seed=seed,
        )
