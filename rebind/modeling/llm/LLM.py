from abc import  abstractmethod
from typing import  Optional
import torch.nn as nn


class LLM(nn.Module):
    """Abstract base class for language models"""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate text based on the input prompts

        Args:
            system_prompt: System-level instructions
            user_prompt: User query/input
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate (None for model default)

        Returns:
            Generated text response
        """
        pass
