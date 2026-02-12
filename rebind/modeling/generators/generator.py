from abc import abstractmethod
from typing import List
import torch.nn as nn
from PIL import Image


class T2IGenerator(nn.Module):
    """Abstract base class for text-to-image generators"""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        prompts: List[str],
        num_images: int = 1,
        size: str = "1024x1024",
        seed: int = -1,
    ) -> List[Image.Image]:
        """Generate images from text prompt"""
