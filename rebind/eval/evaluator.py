from abc import ABC, abstractmethod

import torch
import logging

from eval.eval import parse_yes_no

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import json


class QA_Evaluator(ABC):
    """Abstract base class for QA-based image evaluators (LLaVA, InternVL, GPT4V etc.)"""

    def __init__(
        self,
        device: str = "cuda",
        quantization: bool = False,
    ):
        """Initialize the evaluator.

        Args:
            model: The model to use for evaluation
            processor: The processor/tokenizer for the model
            device: Device to run model on
            quantization: Whether model is quantized
        """
        self.model = None
        self.processor = None
        self.name = None
        self.device = device
        self.quantization = quantization

        if not self.quantization and self.model is not None:
            self.model.to(self.device)

    @abstractmethod
    def get_chat_response(
        self,
        image,
        question,
        max_new_tokens=5,
        temperature=0,
    ) -> str:
        """Get response from model for a single question.

        Args:
            image: Input image tensor
            question: Question string
            max_new_tokens: Max number of tokens to generate
            temperature: Sampling temperature

        Returns:
            Model's response string
        """
        pass

    @abstractmethod
    def get_options_logprobs(self, image, question, options, n_tokens) -> torch.Tensor:
        """Get log probabilities for answer options.

        Args:
            image: Input image
            question: Question string
            options: List of possible answer options
            n_tokens: Number of tokens to consider

        Returns:
            Log probabilities for each option
        """
        pass

    @torch.no_grad()
    def qa_score(
        self,
        image,
        eval_triplet,
        categories,
        mode="weighted_average",
        qa_eval_json_path=None,
    ):
        """Compute QA-based score for an image."""
        QA_by_category = self.get_qa_answers(image, qa_eval_json_path)
        assert type(categories) == list, "categories must be a list"
        if mode == "weighted_average":
            score, score_per_category = self.weighted_average(
                QA_by_category, categories
            )
            return score, score_per_category, QA_by_category
        else:
            raise ValueError(f"QA eval mode not registered! got: {mode}")

    @torch.no_grad()
    def get_qa_answers(self, image, qa_eval_json_path):
        """Compute QA-based score for an image.

        Args:
            image: Input image
            triplet: Relation triplet string
            prompt: Prompt string

        Returns:
            Dict of QA results by category
        """

        assert (
            qa_eval_json_path.exists()
        ), f"Evaluation JSON file not found: {qa_eval_json_path}"

        with open(qa_eval_json_path, "r") as f:
            questions_by_category = json.load(f)

        QA_by_category = {}

        for category in questions_by_category:
            QA_by_category[category] = {
                "weight": questions_by_category[category]["weight"],
                "questions": {},
            }

            for question in questions_by_category[category]["questions"]:
                response = self.get_chat_response(image, question)
                response_processed = parse_yes_no(response) == "yes"

                QA_by_category[category]["questions"][question] = (
                    "yes" if response_processed else "no"
                )

        return QA_by_category

    def weighted_average(
        self,
        QA_by_category,
        categories,
    ):
        total_score = 0
        total_weight = sum(
            [float(QA_by_category[category]["weight"]) for category in categories]
        )
        score_per_category = {}
        for category in categories:
            category_score = 0
            for question in QA_by_category[category]["questions"]:
                if QA_by_category[category]["questions"][question] == "yes":
                    category_score += 1
                elif QA_by_category[category]["questions"][question] == "no":
                    category_score += 0
                elif type(QA_by_category[category]["questions"][question]) == float:
                    category_score += QA_by_category[category]["questions"][question]
            category_score /= len(QA_by_category[category]["questions"])
            total_score += category_score * float(QA_by_category[category]["weight"])
            score_per_category[category] = category_score
        return max(0.001, total_score / total_weight), score_per_category

   