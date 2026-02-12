from typing import Optional
from openai import OpenAI
import torch.nn as nn
import logging

from modeling.llm.LLM import LLM
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPT4(LLM):
    """GPT-4 language model implementation"""

    def __init__(self, model: str = "gpt-4"):
        super().__init__()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def forward(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate text using GPT-4

        Args:
            system_prompt: System-level instructions
            user_prompt: User query/input
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response

        Raises:
            Exception: If API call fails
        """
        try:
            print("HIHIIII")
            print(user_prompt)
            print(system_prompt)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error in GPT-4 API call: {str(e)}")
            raise


class GPT_reasoning(LLM):
    """GPT-4 reasoning model implementation"""

    def __init__(self, model: str = "gpt-4"):
        super().__init__()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
