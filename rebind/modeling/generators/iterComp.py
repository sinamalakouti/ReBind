from typing import List
import logging
from PIL import Image
import torch
from diffusers import DiffusionPipeline
from modeling.generators.generator import T2IGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IterComp(T2IGenerator):
    """Wrapper around Stable Diffusion 3 implementation"""

    def __init__(
        self,
        model_id: str = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        logger.info(f"Using device: {self.device}")

        self.pipe = DiffusionPipeline.from_pretrained(
            "comin/IterComp", torch_dtype=torch.float16, use_safetensors=True
        ).to(self.device)
        # Enable memory optimization
        if self.device == "cuda":
            self.pipe.enable_model_cpu_offload()

        # Default negative prompt for better quality
        self.negative_prompt = (
            "distorted, deformed, disfigured, bad anatomy, extra limbs, "
            "blurry, low quality, low resolution, watermark, text, logo, "
            "duplicate, poorly drawn, mutated, mutilated"
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
        Generate images using SD3

        Args:
            prompt: Text prompt
            num_images: Number of images to generate
            size: Image size (e.g., "1024x1024")
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt

        Returns:
            List of generated images
        """
        logger.info(f"GENERATING IMAGES IN SIZE OF {size}")

        if seed > 0:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = torch.Generator(device=self.device)
        generated_images = []
        width, height = map(int, size.split("x"))
        try:
            for i in range(0, num_images):
                # Parse size
                output = self.pipe(
                    prompt=prompts[i],
                    negative_prompt=self.negative_prompt,
                    num_images_per_prompt=1,
                    num_inference_steps=num_inference_steps,
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
        Generate images using SD3 (implements T2IGenerator interface)
        """
        return self.generate_images(
            prompts=prompts, num_images=num_images, size=size, seed=seed
        )
