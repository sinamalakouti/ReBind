from io import BytesIO
import logging
from typing import List
from PIL import Image
from openai import OpenAI
import requests
import os

from modeling.generators.generator import T2IGenerator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Dalle3(T2IGenerator):
    """Wrapper around existing DALL-E 3 implementation"""

    def __init__(self):
        super().__init__()

    def generate_images(
        self,
        prompts: List[str],
        num_images: int = 1,
        size: str = "1024x1024",
        model: str = "dall-e-3",
        seed: int = -1,
    ):
        """
        Generate images using DALL-E and save them to disk
        """

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Setudp output directory

        generated_images = []
        num_failed = 0
        logger.info(f"Generating {num_images} images")
        for i in range(0, num_images):
            try:
                # Generate image
                if seed > 0:
                    response = client.images.generate(
                        model=model,
                        prompt=prompts[i],
                        size=size,
                        quality="standard",
                        n=1,
                    )
                else:
                    response = client.images.generate(
                        model=model,
                        prompt=prompts[i],
                        size=size,
                        quality="standard",
                        n=1,
                    )
                logger.info(f"response: {response}")
                # Get image URL
                image_url = response.data[0].url
                response = requests.get(image_url)

                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content))
                    generated_images.append(image)
                    num_failed = 0
                else:
                    logger.error(f"Failed to download image {i+1}")
                    num_failed += 1
                    if num_failed > 3:
                        raise Exception(
                            f"Failed to download image {i+1}, prompt: {prompts[i]}"
                        )
                    i -= 1

            except Exception as e:
                logger.error(f"Error generating image {i+1}: {str(e)}")
                num_failed += 1
                if num_failed > 3:
                    raise Exception(
                        f"Failed to generate image {i+1}, prompt: {prompts[i]}"
                    )
                i -= 1
                continue
        logger.info(f"Generated {len(generated_images)} images")
        return generated_images

    def forward(
        self,
        prompts: List[str],
        num_images: int = 1,
        size: str = "1024x1024",
        seed: int = -1,
    ) -> List[Image.Image]:
        """
        Generate images using existing DALL-E implementation
        """
        return self.generate_images(
            prompts=prompts,
            num_images=num_images,
            size=size,
            model="dall-e-3",
            seed=seed,
        )
