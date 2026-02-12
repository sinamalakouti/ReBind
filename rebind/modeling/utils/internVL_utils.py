import numpy as np
from sympy import O
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    if type(image_file) == str:
        image = Image.open(image_file).convert("RGB")
    else:
        image = image_file
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def get_model_internVL(model_name="OpenGVLab/InternVL2-8B", quantization=False):
    # If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
    quantization = False  # todo currently I get error I need to see if I can install more uptodate version of bitsandbytes
    path = model_name
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        load_in_8bit=quantization,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, use_fast=False
    )

    return tokenizer, model


@torch.no_grad()
def get_chat_response(
    model, tokenizer, image, question, max_new_tokens=5, temperature=0
):
    # set the max number of tiles in `max_num`
    pixel_values = load_image(image).to(model.device).to(torch.bfloat16)
    # pixel_values = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
    generation_config = dict(
        max_new_tokens=max_new_tokens,
        output_logits=True,
        output_scores=True,
        temperature=temperature,
    )
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    # print("scores: ", response.scores)
    # print("sequence: ", response.sequences)
    # print(f'User: {question}\nAssistant: {response}')
    return (
        response.lower().split("assistant:")[1]
        if "assistant" in response.lower()
        else response
    )


def prepare_input(
    tokenizer,
    pixel_values,
    question,
    generation_config,
    history=None,
    return_history=False,
    num_patches_list=None,
    IMG_START_TOKEN="<img>",
    IMG_END_TOKEN="</img>",
    IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
    verbose=False,
):
    from internvl.conversation import get_conv_template

    if history is None and pixel_values is not None and "<image>" not in question:
        question = "<image>\n" + question

    if num_patches_list is None:
        num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
    assert pixel_values is None or len(pixel_values) == sum(num_patches_list)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    img_context_token_id = img_context_token_id
    template = get_conv_template(template)
    template.system_message = self.system_message
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
    history = [] if history is None else history
    for old_question, old_answer in history:
        template.append_message(template.roles[0], old_question)
        template.append_message(template.roles[1], old_answer)
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()
    if verbose and pixel_values is not None:
        image_bs = pixel_values.shape[0]
        print(f"dynamic ViT batch size: {image_bs}")
    for num_patches in num_patches_list:
        image_tokens = (
            IMG_START_TOKEN
            + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
            + IMG_END_TOKEN
        )
        query = query.replace("<image>", image_tokens, 1)
    model_inputs = tokenizer(query, return_tensors="pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = model_inputs["input_ids"].to(device)
    attention_mask = model_inputs["attention_mask"].to(device)
    generation_config["eos_token_id"] = eos_token_id
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "generation_config": generation_config,
    }


@torch.no_grad()
def get_options_logprobs(model, tokenizer, image, question, options, n_tokens=5):
    """Get log probabilities for each option using InternVL model."""

    # Load and preprocess image
    pixel_values = load_image(image).to(model.device).to(torch.bfloat16)

    # Format question with options
    options_text = ", ".join(options)
    full_question = f"{question} Options: {options_text}. Answer:"
    # Generation config for InternVL
    generation_config = dict(
        max_new_tokens=n_tokens,
        # num_beams=1,
        # do_sample=True,
        output_scores=True,
        use_return_dict=True,
    )

    # Get response with scores
    outputs = model.chat(tokenizer, pixel_values, full_question, generation_config)

    # Process the outputs
    results = {
        "text": outputs.sequences if hasattr(outputs, "sequences") else str(outputs),
        "scores": {},
    }

    # Get scores for each option
    for option in options:
        # InternVL tokenizer doesn't have separate tokenizer attribute
        option_tokens = tokenizer.encode(option, add_special_tokens=False)
        sequence_log_prob = 0.0

        # Get scores if available
        if hasattr(outputs, "scores"):
            all_scores = outputs.scores
            print("scores shape: ", all_scores.shape)
            for i, token_id in enumerate(option_tokens):
                if i < len(all_scores):
                    token_scores = all_scores[i]
                    token_log_probs = torch.nn.functional.log_softmax(
                        token_scores, dim=-1
                    )
                    sequence_log_prob += token_log_probs[0, token_id].item()
        else:
            raise ValueError("No scores found in the outputs")

        # Store results
        results["scores"][option] = {
            "log_prob": (
                sequence_log_prob / len(option_tokens)
                if len(option_tokens) > 0
                else float("-inf")
            ),
            "num_tokens": len(option_tokens),
        }

        # Debug print
        print(f"\nOption: '{option}'")
        print(f"Tokens: {tokenizer.convert_ids_to_tokens(option_tokens)}")
        print(f"Log prob: {results['scores'][option]['log_prob']:.4f}")

    return results


def parse_yes_no(response):
    if "yes" in response.lower():
        return "yes"
    elif "no" in response.lower():
        return "no"
    else:
        raise ValueError(
            f"EXPECTED: Response must contain either 'yes' or 'no'. OBSERVED: {response}"
        )
