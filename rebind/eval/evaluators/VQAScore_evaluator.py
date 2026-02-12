import torch
import logging
from eval.evaluator import QA_Evaluator
from prompts.prompt_utils import get_base_prompt_from_action_triplet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import json
import os

import t2v_metrics


class VQAScore_Evaluator(QA_Evaluator):
    """Abstract base class for QA-based image evaluators (LLaVA, InternVL, GPT4V etc.)"""

    def __init__(
        self,
        device: str = "cuda",
        quantization: bool = False,
        # VQA_model_name="clip-flant5-xxl",
        VQA_model_name="gpt-4o",
    ):
        """Initialize the evaluator.

        Args:
            model: The model to use for evaluation
            processor: The processor/tokenizer for the model
            device: Device to run model on
            quantization: Whether model is quantized
        """
        self.VQA_model_name = VQA_model_name
        if self.VQA_model_name == "gpt-4o":
            self.model = t2v_metrics.VQAScore(
                model=self.VQA_model_name, openai_key=os.getenv("OPENAI_API_KEY"), top_logprobs=20
            )
        else:
            self.model = t2v_metrics.VQAScore(model=self.VQA_model_name)
        self.processor = None
        self.device = device
        self.quantization = quantization

        self.name = "VQAScore_{}".format(self.VQA_model_name)

        if not self.quantization and self.model is not None:
            self.model.to(self.device)

    def get_chat_response(
        self,
        image,
        question,
        max_new_tokens=5,
        temperature=0,
    ) -> str:
        assert False, "VQAScore_Evaluator does not support get_chat_response"

    def get_options_logprobs(self, image, question, options, n_tokens):
        """Get log probabilities for answer options.

        Args:
            image: Input image
            question: Question string
            options: List of possible answer options
            n_tokens: Number of tokens to consider

        Returns:
            Log probabilities for each option
        """
        assert False, "VQAScore_Evaluator does not support get_options_logprobs"

    @torch.no_grad()
    def get_qa_answers(self, image, qa_eval_json_path):
        """Get QA answers for each category from the model.

        Returns dict in format:
        {
            "category": {
                "weight": float,
                "questions": {
                    "question_text": score,  # Instead of "yes"/"no", keep the raw score
                    ...
                }
            }
        }
        """
        assert (
            qa_eval_json_path.exists()
        ), f"Evaluation JSON file not found: {qa_eval_json_path}"

        with open(qa_eval_json_path, "r") as f:
            qa_json = json.load(f)

        # Get all statements for batch processing
        all_statements = []
        for category in qa_json.values():
            all_statements.extend(category["statements"])

        # Get scores for all statements at once
        scores = self.model(images=image, texts=all_statements).squeeze(0)

        # Create output dictionary
        QA_by_category = {}
        current_idx = 0

        for category, content in qa_json.items():
            statements = content["statements"]
            questions = content["questions"]
            num_items = len(statements)

            QA_by_category[category] = {"weight": content["weight"], "questions": {}}

            # Map the scores to questions
            category_scores = scores[current_idx : current_idx + num_items]
            for question, score in zip(questions, category_scores):
                QA_by_category[category]["questions"][question] = float(score.item())

            current_idx += num_items

        return QA_by_category

    @torch.no_grad()
    def get_VQA_score(self, image, eval_triplet):
        """Get QA answers for each category from the model.

        Returns dict in format:
        {
            "category": {
                "weight": float,
                "questions": {
                    "question_text": score,  # Instead of "yes"/"no", keep the raw score
                    ...
                }
            }
        }
        """
        # Get all statements for batch processing
        statement = get_base_prompt_from_action_triplet(
            eval_triplet, prefix="", add_determiners=False
        )

        # Get scores for all statements at once
        scores = self.model(images=image, texts=[statement]).squeeze(0)

        return scores[0].item()
