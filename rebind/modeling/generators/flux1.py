from typing import List
import logging
from PIL import Image
import torch
from diffusers import DiffusionPipeline
from modeling.generators.generator import T2IGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FLUX(T2IGenerator):
    """Wrapper around FLUX implementation"""

    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.1-dev",
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        logger.info(f"Using device: {self.device}")

        # Load model
        self.pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
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
        size: str = "512x512",
        num_inference_steps: int = 50,
        seed: int = -1,
    ) -> List[Image.Image]:
        """
        Generate images using FLUX

        Args:
            prompt: Text prompt
            num_images: Number of images to generate
            size: Image size (e.g., "1024x1024")
            num_inference_steps: Number of denoising steps
            seed: Random seed for generation

        Returns:
            List of generated images
        """

        if seed > 0:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = torch.Generator(device=self.device)

        generated_images = []
        width, height = map(int, size.split("x"))
        try:
            for i in range(0, num_images):
                output = self.pipe(
                    prompt=prompts[i],
                    num_inference_steps=num_inference_steps,
                    width=width,
                    height=height,
                    generator=generator,
                ).images[0]

                # Convert to PIL images if needed
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
        Generate images using FLUX (implements T2IGenerator interface)
        """
        return self.generate_images(
            prompts=prompts, num_images=num_images, size=size, seed=seed
        )
