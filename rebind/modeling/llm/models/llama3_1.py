from typing import Optional
import transformers
import torch
import torch.nn as nn
import logging
from modeling.llm.LLM import LLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLama3_1(LLM):
    """Llama 3.1 language model implementation"""

    def __init__(self, model_id: str = "meta-llama/Llama-3.1-8B-Instruct"):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

    def forward(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = 256,
    ) -> str:
        """
        Generate text using Llama 3.1

        Args:
            system_prompt: System-level instructions
            user_prompt: User query/input
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response

        Raises:
            Exception: If model generation fails
        """
        try:

            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": user_prompt},
            ]

            outputs = self.pipeline(
                messages,
                max_new_tokens=max_tokens,
            )

            response = outputs[0]["generated_text"][-1]

            return response

        except Exception as e:
            logger.error(f"Error in Llama 3.1 generation: {str(e)}")
            raise
