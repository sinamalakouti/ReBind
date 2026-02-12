import torch
from diffusers import StableDiffusion3Pipeline
from attention_map_diffusers import attn_maps, init_pipeline, save_attention_maps
import re

# rolebench data and templates
RELATIONS = [
    "chasing",
    "riding",
    "throwing",
    "holding",
    "following",
    "feeding",
    "pulling",
    "lifting",
    "carrying",
    "kissing",
]

RELATION_BASE_PROMPT_TEMPLATE = {
    "chasing": "{subject} chasing {object}. Chaser, chased",
    "feeding": "{subject} feeding food to {object}. Feeder, fed",
    "riding": "{subject} riding on {object}. Rider, ridden",
    "throwing": "{subject} throwing a ball to {object}. Thrower, thrown",
    "pulling": "{subject} pulling {object}. Puller, pulled",
    "lifting": "{subject} lifting {object}. Lifter, lifted",
    "carrying": "{subject} carrying {object}. Carrier, carried",
    "holding": "{subject} holding {object}. Holder, held",
    "following": "{subject} following {object}. Follower, followed",
}

rolebench_data = {
    "chasing": {
        "frequent": "cat_chasing_mouse",
        "rare": "mouse_chasing_cat",
    },
    "riding": {
        "frequent": "astronaut_riding_horse",
        "rare": "horse_riding_astronaut",
    },
    "throwing": {
        "frequent": "boy_throwing_puppy",
        "rare": "puppy_throwing_boy",
    },
    "holding": {
        "frequent": "grandpa_holding_doll",
        "rare": "doll_holding_grandpa",
    },
    "following": {
        "frequent": "lion_following_cow",
        "rare": "cow_following_lion",
    },
    "feeding": {
        "frequent": "woman_feeding_baby",
        "rare": "baby_feeding_woman",
    },
    "kissing": {
        "frequent": "mother_kissing_baby",
        "rare": "baby_kissing_mother",
    },
    "pulling": {
        "frequent": "man_pulling_dog",
        "rare": "dog_pulling_man",
    },
    "lifting": {
        "frequent": "zoo_trainer_lifting_monkey",
        "rare": "monkey_lifting_zoo_trainer",
    },
    "carrying": {
        "frequent": "fireman_carrying_scientist",
        "rare": "scientist_carrying_fireman",
    },
}


def parse_action_triplet(subject_rel_obj, novel_relation=False):
    pattern = f"^(.+)_({('|'.join(RELATIONS))})_(.+)$"
    match = re.match(pattern, subject_rel_obj)
    if not match:
        if not novel_relation:
            raise ValueError(f"No valid relation found in: {subject_rel_obj}")
        else:
            raise Warning(f"No valid relation found in: {subject_rel_obj}")
    subject, relation, obj = match.groups()
    subject = subject.replace("_", " ").lower()
    obj = obj.replace("_", " ").lower()
    return subject.lower(), relation.lower(), obj.lower()


def get_prompts_from_rolebench():
    """Extract rare and frequent prompts from rolebench data."""
    prompts = []
    for relation, data in rolebench_data.items():
        # Get the base template for this relation
        template = RELATION_BASE_PROMPT_TEMPLATE[relation]

        # Process frequent case

        frequent_prompt = get_base_prompt_from_action_triplet(data["frequent"])
        prompts.append(("frequent", relation, data["frequent"], frequent_prompt))

        # Process rare case
        rare_prompt = get_base_prompt_from_action_triplet(data["rare"])
        prompts.append(("rare", relation, data["rare"], rare_prompt))

    return prompts


def get_base_prompt_from_action_triplet(
    action_triplet, prefix="A photo of ", add_determiners=True, novel_relation=False
):
    sub, relation, obj = parse_action_triplet(
        action_triplet, novel_relation=novel_relation
    )
    if "_" in sub:
        sub = sub.replace("_", " ")
    if "_" in obj:
        obj = obj.replace("_", " ")
    if add_determiners:
        sub = "one " + sub
        obj = "one " + obj
    if relation not in RELATION_BASE_PROMPT_TEMPLATE:
        raise ValueError(f"Unknown relation: {relation}")
    return prefix + RELATION_BASE_PROMPT_TEMPLATE[relation].format(
        subject=sub, object=obj
    )


def main():
    # Model configuration
    MODEL_NAME = "stabilityai/stable-diffusion-3-medium-diffusers"
    MODEL_ID = "sd3-medium"

    # Initialize pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16
    )
    pipe = pipe.to("cuda")

    # Replace modules and Register hook
    pipe = init_pipeline(pipe)

    # Get prompts from rolebench
    all_prompts = get_prompts_from_rolebench()

    # Process each prompt individually to avoid memory issues
    for prompt_type, relation, triplet, prompt in all_prompts:
        print(f"\nProcessing {prompt_type} {relation} prompt: {prompt}")

        # Generate 5 different samples
        for sample_idx in range(1, 6):
            print(f"Generating sample {sample_idx}/5")

            # Set output directory structure with model name
            output_dir = f"outputs/{MODEL_ID}/{relation}/{triplet}/{sample_idx}"
            os.makedirs(output_dir, exist_ok=True)

            # Generate image with different seed for each sample
            generator = torch.Generator(device="cuda").manual_seed(42 + sample_idx)
            images = pipe(
                [prompt],
                num_inference_steps=15,
                guidance_scale=4.5,
                generator=generator,
                height=512,
                width=512,
            ).images

            # Save generated image
            images[0].save(f"{output_dir}/generated_image.png", quality=100)

            # Save attention maps
            attn_map_dir = f"{output_dir}/attention_maps"
            save_attention_maps(
                attn_maps,
                pipe.tokenizer,
                [prompt],
                base_dir=attn_map_dir,
                unconditional=True,
            )

            # Clear CUDA cache after each generation
            torch.cuda.empty_cache()


if __name__ == "__main__":
    import os

    os.makedirs("outputs", exist_ok=True)
    main()
