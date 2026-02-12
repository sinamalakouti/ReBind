def get_evaluator_class(
    model_name: str, device: str = "cuda"
):
    from eval.evaluators.VQAScore_evaluator import VQAScore_Evaluator
    from eval.evaluators.internVL_evaluator import InternVL_Evaluator

    """Get the appropriate evaluator class based on model name.

    Args:
        model_name: Name of the model (e.g., 'internvl', 'llava')

    Returns:
        Evaluator class corresponding to the model name

    Raises:
        ValueError: If model_name is not supported
    """
    model_name = model_name.lower()
    avialable_evaluators = ["internvl", "vqascore_gpt-4o", "vqascore_clip-flant5-xxl"]
    if model_name == "internvl":
        return InternVL_Evaluator(device=device)
    elif model_name == "vqascore_gpt-4o":
        return VQAScore_Evaluator(
            VQA_model_name="gpt-4o"
        )
    elif model_name == "vqascore_clip-flant5-xxl":
        return VQAScore_Evaluator(VQA_model_name="clip-flant5-xxl", device=device)
    else:
        raise ValueError(
            f"Unsupported model: {model_name}. Available models: {list(avialable_evaluators)}"
        )
