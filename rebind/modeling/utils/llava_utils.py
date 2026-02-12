from tempfile import tempdir
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
from tqdm import tqdm
import torch



def get_image(image_path):
    image = Image.open(image_path)
    return image


@torch.no_grad()
def get_batch_response(model, processor, images, question):
    final_images = []
    for image in images:
        if type(image) == str:
            image = get_image(image)
        final_images.append(image)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{question}"},
                {"type": "image"},
            ],
        },
    ]

    prompts = []
    for i in range(len(images)):
        prompts.append(
            processor.apply_chat_template(conversation, add_generation_prompt=True)
        )
    inputs = processor(
        images=final_images, text=prompts, padding=True, return_tensors="pt"
    ).to(model.device)

    generate_ids = model.generate(**inputs, max_new_tokens=5)
    responses = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    processed_responses = [response.split("ASSISTANT:")[1] for response in responses]
    return processed_responses


@torch.no_grad()
def get_options_logprobs_all_vocab(
    model, processor, image, question, options, n_tokens=5
):
    if type(image) == str:
        image = get_image(image)

    # Format question with options
    options_text = ", ".join(options)
    print("question : ", question)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{question}"},
                {"type": "image"},
            ],
        },
    ]

    # print("conversation : ", conversation)

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

    # Generate with output scores for multiple tokens
    outputs = model.generate(
        **inputs,
        max_new_tokens=n_tokens,  # Increased to handle multi-token words
        num_beams=1,
        do_sample=True,
        output_scores=True,
        return_dict_in_generate=True,
    )

    # Get all scores for the generated sequence
    all_scores = torch.stack(
        outputs.scores
    )  # Shape: [num_tokens, batch_size, vocab_size]

    # Get probabilities for each option
    results = {
        "text": processor.decode(outputs.sequences[0], skip_special_tokens=True),
        "scores": {},
    }
    for option in options:
        option_tokens = processor.tokenizer.encode(option, add_special_tokens=False)
        sequence_log_prob = 0.0
        for i, token_id in enumerate(option_tokens):
            if i < len(all_scores):
                token_scores = all_scores[i]
                token_log_probs = torch.nn.functional.log_softmax(token_scores, dim=-1)
                sequence_log_prob += token_log_probs[0, token_id].item()
        results["scores"][option] = {
            "log_prob": sequence_log_prob / len(option_tokens),
            "num_tokens": len(option_tokens),
        }
    return results


@torch.no_grad()
def get_options_logprobs(model, processor, image, question, options, n_tokens=5):
    if type(image) == str:
        image = get_image(image)

    # Format question with options
    # options_text = ", ".join(options)
    # full_question = f"{question} Options: {options_text}. Answer:"
    print("question : ", question)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{question}"},
                {"type": "image"},
            ],
        },
    ]

    # print("conversation : ", conversation)

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=False)
    print("prompt : ", prompt)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=n_tokens,
        num_beams=1,
        do_sample=False,
        output_logits=True,
        output_scores=True,
        return_dict_in_generate=True,
        temperature=0.01,
    )

    all_scores = torch.stack(outputs.logits).to(
        torch.float32
    )  # [num_tokens, batch_size, vocab_size]

    # First compute average token scores for each option
    option_avg_scores = []
    for option in options:
        option_tokens = processor.tokenizer.encode(option, add_special_tokens=False)
        token_scores = []
        for i, token_id in enumerate(option_tokens):
            if i < len(all_scores):
                # Get raw score for this token
                score = all_scores[i, 0, token_id]
                token_scores.append(score)
        # Average score across tokens for this option
        option_avg_scores.append(torch.mean(torch.stack(token_scores)))

    # Now compute softmax over just the options
    option_scores = torch.stack(option_avg_scores)
    option_log_probs = torch.log_softmax(option_scores, dim=0)

    results = {
        "text": processor.decode(outputs.sequences[0], skip_special_tokens=True),
        "scores": {
            option: {
                "log_prob": log_prob.item(),
                "num_tokens": len(
                    processor.tokenizer.encode(option, add_special_tokens=False)
                ),
            }
            for option, log_prob in zip(options, option_log_probs)
        },
    }
    # print("logprobs for relations: ", results["scores"])
    return results


@torch.no_grad()
def get_chat_response(
    model,
    processor,
    image,
    question,
    max_new_tokens=5,
    temperature=0,
):
    if type(image) == str:
        image = get_image(image)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{question}"},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs, max_new_tokens=max_new_tokens, temperature=temperature
    )
    response = processor.decode(output[0], skip_special_tokens=True)

    return response.split("ASSISTANT:")[1]


def get_model(model_name, quantization):
    processor = LlavaNextProcessor.from_pretrained(model_name)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        load_in_8bit=True if quantization else False,
        # low_cpu_mem_usage=True,
        # device_map="auto",
        attn_implementation="flash_attention_2",
    )
    return processor, model


def parse_yes_no(response):
    if "yes" in response.lower():
        return "yes"
    elif "no" in response.lower():
        return "no"
    else:
        return "no"
        # raise ValueError(f"EXPECTED: Response must contain either 'yes' or 'no'. OBSERVED: {response}")
