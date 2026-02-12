
import json
import argparse
import logging
import csv
from typing import List
from PIL import Image


from tqdm import tqdm
from configs.config import load_config, log_config, merge_configs
from dataset.paths import get_data_dir
from eval.evaluators import get_evaluator_class
from modeling.generators import flux1
from modeling.generators.auraflow2 import AuraFlow2
from modeling.generators.generator import T2IGenerator
from modeling.generators.iterComp import IterComp
from modeling.generators.sd3 import SD3
from modeling.generators.sdxl import SDXL
from modeling.generators.dalle import Dalle3
from modeling.llm.models.gpt4 import GPT4
from modeling.llm.models.llama3_1 import LLama3_1
from prompts.prompt_utils import get_base_prompt_from_action_triplet
from prompts.templates.llm_templates import create_prompt_expander_template
from dataset.rolebench import (
    RELATIONS,
    get_evaluation_triplets,
    reverse_triplet, 
    rolebench_data,
)
from eval.eval import evaluate, generate_qa_eval_json
from prompts.templates.llm_templates import (
    get_intermediate_active_passive_triplet_template,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _get_expanded_prompt(
    original_triplet, llm_model_name, num_images, save_dir, cfg
) -> List[str]:
    """Get expanded prompt from CSV if exists, otherwise generate new one

    Args:
        original_prompt: Original prompt to expand
        llm_model: Name of the LLM model to use
        csv_path: Path to the mapping CSV file

    Returns:
        str: Expanded prompt
    """
    original_prompt = get_base_prompt_from_action_triplet(original_triplet)
    # set prompt file path
    prompt_file_path = save_dir / f"{original_triplet}.txt"
    if prompt_file_path.exists():
        with open(prompt_file_path, "r") as f:
            # Read line by line and strip whitespace
            expanded_prompts = [line.strip() for line in f.readlines() if line.strip()]
    else:
        expanded_prompts = []

    if len(expanded_prompts) == num_images:
        return expanded_prompts
    else:
        num_images = num_images - len(expanded_prompts)

    # If not found or error, generate new expanded prompt
    logger.info(f"Generating new expanded prompt for: {original_triplet}")
    if (
        "gpt" in llm_model_name.lower()
        or "o3" in llm_model_name.lower()
        or "o1" in llm_model_name.lower()
    ):
        LLM = GPT4(model=llm_model_name)
    elif "llama3_1" in llm_model_name.lower():
        LLM = LLama3_1()

    expander_template = create_prompt_expander_template()
    for i in tqdm(range(num_images)):
        expanded_prompt = LLM(
            system_prompt=expander_template["system_prompt"],
            user_prompt=expander_template["user_prompt"].format(prompt=original_prompt),
        )
        expanded_prompts.append(expanded_prompt)

    # Write each prompt on a new line
    with open(prompt_file_path, "w") as f:
        for text in expanded_prompts:
            f.write(text + "\n")

    return expanded_prompts


def generate_images(
    prompts: List[str],
    generator: T2IGenerator,
    num_images: int = 5,
    seed: int = -1,
):
    """Generate base images for a given triplet using specified generator"""

    # Setup CSV for mapping

    # Generate missing images
    # num_to_generate = num_images - len(existing)
    new_images = generator(
        prompts=prompts, num_images=num_images, size="1024x1024", seed=seed
    )

    generated_images = []
    # Save new images and update CSV
    for i, img in enumerate(new_images):
        # Resize image to 512x512
        if isinstance(img, Image.Image):
            img = img.resize((512, 512), Image.Resampling.LANCZOS)

            # Save image
            generated_images.append(img)

        logger.info(f"Generated {len(new_images)} new images")

    return generated_images


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--T2I",
        type=str,
        choices=["sd3", "sd3-5", "sdxl", "dall-e-3", "flux1", "auraflow2", "itercomp"],
        required=True,
        help="Which generator to use",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=20,
        help="Number of images to generate per prompt",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Seed for image generation",
    )
    parser.add_argument(
        "--expand_prompt",
        action="store_true",
        help="Expand the prompt to include more details",
    )
    parser.add_argument(
        "--base_prompt",
        action="store_true",
        help="Generate base prompt images",
    )
    parser.add_argument(
        "--evaluator_model",
        type=str,
        required=False,
        help="Evaluator to use for evaluation",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["qa_score"],
        help="Metrics to use for evaluation. Choose one or more from: qa_score, rel_score, entropy_score, vqascore",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the generated images",
    )
    parser.add_argument(
        "--generate_intermediate",
        action="store_true",
        help="Generate intermediate triplets and their images",
    )
    return parser.parse_args()


def get_generator(cfg):
    # Initialize generator
    if cfg["T2I"].lower() == "sd3":
        generator = SD3("stabilityai/stable-diffusion-3-medium-diffusers")
    elif cfg["T2I"].lower() == "sd3-5":
        generator = SD3("stabilityai/stable-diffusion-3.5-medium")
    elif cfg["T2I"].lower() == "sdxl":
        generator = SDXL()
    elif cfg["T2I"].lower() == "dall-e-3":
        generator = Dalle3()
    elif cfg["T2I"].lower() == "flux1":
        generator = flux1.FLUX()
    elif cfg["T2I"].lower() == "auraflow2":
        generator = AuraFlow2()
    elif cfg["T2I"].lower() == "itercomp":
        generator = IterComp()
    else:
        raise ValueError(f"Unknown generator: {cfg['T2I']}")

    logger.info(f"Initialized {cfg['T2I']} generator")
    return generator


def process_triplet(triplet, relation, triplet_dir, prompt_dir, image_dir, cfg):
    base_image_dir = image_dir / "images"
    expanded_image_dir = image_dir / "expanded_images"
    base_image_dir.mkdir(parents=True, exist_ok=True)
    eval_json_path = prompt_dir / "evaluation_qa.json"
    if cfg["expand_prompt"]:
        expanded_image_dir.mkdir(parents=True, exist_ok=True)
    generator = None
    # Get base prompt
    base_prompt = get_base_prompt_from_action_triplet(
        triplet, prefix="A photorealistic image of "
    )
    logger.info(f"Generating images for prompt: {base_prompt}")

    if cfg["base_prompt"]:
        # Check existing images
        existing = list(base_image_dir.glob("image_*.png"))
        if len(existing) < cfg["num_images"]:
            if generator is None:
                generator = get_generator(cfg)
            # Generate base prompt images
            images = generate_images(
                prompts=[base_prompt] * (cfg["num_images"] - len(existing)),
                generator=generator,
                num_images=cfg["num_images"] - len(existing),
                seed=cfg["seed"],
            )

            # Save base images
            for i, img in enumerate(images, start=len(existing)):
                image_name = f"image_{i+1}.png"
                img_path = base_image_dir / image_name
                img.save(img_path)
        else:
            logger.info(f"Skipping generation, already have {len(existing)} images")

    # Handle expanded prompt if requested
    if cfg["expand_prompt"]:
        csv_path = triplet_dir / "expanded_prompt_mapping.csv"

        expanded_prompts = _get_expanded_prompt(
            original_triplet=triplet,
            llm_model_name=cfg["model"]["llm_model"],
            num_images=cfg["num_images"],
            save_dir=prompt_dir,
            cfg=cfg,
        )

        existing = list(expanded_image_dir.glob("image_*.png"))

        if len(existing) < cfg["num_images"]:
            if generator is None:
                generator = get_generator(cfg)
            # Generate expanded prompt images
            assert len(expanded_prompts) - len(existing) == cfg["num_images"] - len(
                existing
            ), f"Number of expanded prompts does not match number of images. Expanded prompts: {len(expanded_prompts)}, new images: {cfg['num_images'] - len(existing)}"
            expanded_images = generate_images(
                prompts=expanded_prompts,
                generator=generator,
                num_images=cfg["num_images"] - len(existing),
                seed=cfg["seed"],
            )

            # Save expanded images and create mapping
            if not csv_path.exists():
                # Create new file with headers
                with open(csv_path, "w", newline="") as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(["image_name", "expanded_prompt"])

            # Open file once for all writes
            with open(csv_path, "a", newline="") as csv_file:
                writer = csv.writer(csv_file)
                for i, img in enumerate(expanded_images, start=len(existing)):
                    image_name = f"image_{i+1}.png"
                    expanded_prompt = expanded_prompts[i]
                    img_path = expanded_image_dir / image_name
                    img.save(img_path)
                    writer.writerow([f"image_{i+1}.png", expanded_prompt])

    if not eval_json_path.exists():
        qa = generate_qa_eval_json(base_prompt)
        try:
            with open(eval_json_path, "w") as f:
                json.dump(qa, f, indent=2)
            logger.info(f"Successfully saved evaluations to {eval_json_path}")
        except Exception as e:
            logger.error(f"Error saving evaluations to file: {str(e)}")


def generate_intermediate_triplets(triplet: str, cfg) -> dict:
    """Generate intermediate triplets using LLM"""
    if "gpt" in cfg["model"]["llm_model"].lower():
        LLM = GPT4(model=cfg["model"]["llm_model"])
    elif "llama3_1" in cfg["model"]["llm_model"].lower():
        LLM = LLama3_1()

    template = get_intermediate_active_passive_triplet_template()
    response = LLM(
        system_prompt=template["system_prompt"],
        user_prompt=template["user_prompt"].format(
            target_triplet=triplet,
            target_prompt=get_base_prompt_from_action_triplet(triplet),
            temperature=0.3,
        ),
    )
    print("RESPONSE: ", response)
    logger.info(f"RESPONSE: {response}")
    # Clean up the response to get only the JSON part
    try:
        # Find the start of the JSON content (first '{')
        json_start = response.find("{")
        # Find the end of the JSON content (last '}')
        json_end = response.rfind("}") + 1
        # Extract just the JSON part
        json_str = response[json_start:json_end]
        return json.loads(json_str)
    except Exception as e:
        logger.error(f"Failed to parse LLM response: {response}")
        raise e


def process_intermediate_triplets(target_triplet, target_prompt_dir, cfg, evaluator):
    """Process intermediate triplets and generate their images"""
    # Generate and save intermediate triplets
    intermediate_path = target_prompt_dir / "intermediate.json"
    if not intermediate_path.exists():
        intermediates = generate_intermediate_triplets(target_triplet, cfg)
        with open(intermediate_path, "w") as f:
            json.dump(intermediates, f, indent=2)
    else:
        with open(intermediate_path, "r") as f:
            intermediates = json.load(f)

    # Process each intermediate triplet
    for intermediate_type in ["active", "passive"]:
        for item in intermediates[target_triplet][intermediate_type]:
            # Get the triplet and its type
            intermediate_triplet = item["triplet"]
            intermediate_triplet_type = item["type"]

            # Use reverse_triplet to get the contrast pair
            intermediate_contrast = reverse_triplet(intermediate_triplet)

            # Process main triplet
            (
                intermediate_triplet_dir,
                intermediate_triplet_prompt_dir,
                intermediate_triplet_image_dir,
            ) = get_data_dir(
                intermediate_triplet, cfg["T2I"], cfg["model"]["llm_model"]
            )
            process_triplet(
                triplet=intermediate_triplet,
                relation=None,
                triplet_dir=intermediate_triplet_dir,
                prompt_dir=intermediate_triplet_prompt_dir,
                image_dir=intermediate_triplet_image_dir,
                cfg=cfg,
            )

            # Process contrast pair
            (
                contrast_dir,
                contrast_prompt_dir,
                contrast_image_dir,
            ) = get_data_dir(
                intermediate_contrast, cfg["T2I"], cfg["model"]["llm_model"]
            )

            # Generate images for contrast pair
            process_triplet(
                triplet=intermediate_contrast,
                relation=None,
                triplet_dir=contrast_dir,
                prompt_dir=contrast_prompt_dir,
                image_dir=contrast_image_dir,
                cfg=cfg,
            )

            image_types = []
            if cfg["expand_prompt"]:
                image_types.append("expanded_images")
            elif cfg["base_prompt"]:
                image_types.append("images")

            # Evaluate after both generations are complete
            for image_type in image_types:
                if cfg["evaluate"]:
                    # Evaluate main triplet
                    if image_type == "expanded_images":
                        main_image_dir = intermediate_triplet_image_dir / image_type
                        main_output_dir = (
                            intermediate_triplet_image_dir / "expanded_results"
                        )
                    elif cfg["base_prompt"]:
                        main_image_dir = intermediate_triplet_image_dir / image_type
                        main_output_dir = intermediate_triplet_image_dir / "results"

                    if main_image_dir.exists():
                        print(f"Evaluating main triplet: {intermediate_triplet}")
                        print("image_dir: ", main_image_dir)
                        print("output_dir: ", main_output_dir)

                        for metric in cfg["metrics"]:
                            # Evaluate with evaluation triplets
                            for eval_triplet in get_evaluation_triplets(
                                intermediate_triplet
                            ):
                                evaluate(
                                    image_dir=main_image_dir,
                                    eval_triplet=eval_triplet,
                                    metric=metric,
                                    evaluator=evaluator,
                                    output_dir=main_output_dir,
                                    cfg=cfg,
                                    check_exists=True,
                                )

                            # Evaluate with contrast pair
                            evaluate(
                                image_dir=main_image_dir,
                                eval_triplet=intermediate_contrast,
                                metric=metric,
                                evaluator=evaluator,
                                output_dir=main_output_dir,
                                cfg=cfg,
                                check_exists=True,
                            )

                    # Evaluate contrast pair
                    contrast_image_dir = contrast_dir / "expanded_images"
                    contrast_output_dir = contrast_dir / "expanded_results"

                    if contrast_image_dir.exists():
                        print(f"Evaluating contrast pair: {intermediate_contrast}")
                        print("image_dir: ", contrast_image_dir)
                        print("output_dir: ", contrast_output_dir)

                        for metric in cfg["metrics"]:
                            # Evaluate with evaluation triplets
                            for eval_triplet in get_evaluation_triplets(
                                intermediate_contrast
                            ):
                                evaluate(
                                    image_dir=contrast_image_dir,
                                    eval_triplet=eval_triplet,
                                    metric=metric,
                                    evaluator=evaluator,
                                    output_dir=contrast_output_dir,
                                    cfg=cfg,
                                    check_exists=True,
                                )

                            # Evaluate with original triplet
                            evaluate(
                                image_dir=contrast_image_dir,
                                eval_triplet=intermediate_triplet,
                                metric=metric,
                                evaluator=evaluator,
                                output_dir=contrast_output_dir,
                                cfg=cfg,
                                check_exists=True,
                            )


def main():
    args = parse_args()
    cfg = load_config("configs/generate_rolebench.ini")
    cfg = merge_configs(cfg, args)
    log_config(cfg)
    if cfg["evaluate"]:
        evaluator_name = cfg["evaluator_model"]
        evaluator = get_evaluator_class(evaluator_name)
    # RELATIONS = ["kissing"]
    for relation in RELATIONS:
        for triplet_type in ["frequent", "rare"]:
            triplet = rolebench_data[relation][triplet_type]
            triplet_dir, prompt_dir, image_dir = get_data_dir(
                triplet, cfg["T2I"], cfg["model"]["llm_model"]
            )

            # Normal triplet processing
            if not cfg["generate_intermediate"]:
                process_triplet(
                    triplet=triplet,
                    relation=relation,
                    triplet_dir=triplet_dir,
                    prompt_dir=prompt_dir,
                    image_dir=image_dir,
                    cfg=cfg,
                )

                if cfg["evaluate"]:
                    image_types = []
                    if cfg["expand_prompt"]:
                        image_types.append("expanded_images")
                    elif cfg["base_prompt"]:

                        image_types.append("images")
                    for image_type in image_types:
                        if image_type == "images":
                            output_dir = image_dir / "results"
                            image_dir = image_dir / image_type

                        else:
                            output_dir = image_dir / "expanded_results"
                            image_dir = image_dir / image_type

                        print("image_dir: ", image_dir)
                        print("output_dir: ", output_dir)
                        for metric in cfg["metrics"]:
                            for eval_triplet in get_evaluation_triplets(triplet):
                                evaluate(
                                    image_dir=image_dir,
                                    eval_triplet=eval_triplet,
                                    metric=metric,
                                    evaluator=evaluator,
                                    output_dir=output_dir,
                                    cfg=cfg,
                                )
            # Generate intermediates for rare triplets
            elif cfg["generate_intermediate"] and triplet_type == "rare":
                process_intermediate_triplets(
                    target_triplet=triplet,
                    target_prompt_dir=prompt_dir,
                    cfg=cfg,
                    evaluator=evaluator,
                )


if __name__ == "__main__":
    main()
