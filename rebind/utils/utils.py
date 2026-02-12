"""
utilities
"""

import logging
import os
import pickle
from PIL import Image
import csv


def create_output_directories(cfg, output_dir, exp_name):
    make_dir(output_dir)
    output_dir = os.path.join(output_dir, exp_name)
    make_dir(output_dir)


def make_dir(directory):
    """
    Creates a directory if it doesn't exist.

    Args:
        directory (str): The path of the directory to be created.
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory '{directory}' created successfully.")
        else:
            print(f"Directory '{directory}' already exists.")
    except Exception as e:
        print(f"An error occurred: {e}")


def get_args():
    pass


def set_manual_seed():
    pass


def save_init_image_score_pairs_initPrompt(
    image_score_paris, init_prompt, generator_model, eval_vlm, n_questions, output_dir
):
    # assert (
    #     len(image_score_paris) == 256
    # ), "Number of pairs should be 256 found {}".format(len(image_score_paris))

    init_prompt = init_prompt.replace(" ", "_")
    generator_model_name = stable_diffusion_to_name(generator_model)
    dir_path = os.path.join(
        output_dir, generator_model_name, init_prompt, f"eval_{eval_vlm}"
    )

    os.makedirs(dir_path, exist_ok=True)

    filename = f"{init_prompt}_numQ_{n_questions}_INIT.pkl"
    with open(os.path.join(dir_path, filename), "wb") as f:
        pickle.dump(image_score_paris, f)


def load_init_image_score_pairs_initPrompt(
    init_prompt, generator_model, eval_vlm, n_questions, output_dir
):
    # Replace spaces with underscores to match the saved filename
    init_prompt = init_prompt.replace(" ", "_")
    generator_model_name = stable_diffusion_to_name(generator_model)

    # Construct the directory path
    dir_path = os.path.join(
        output_dir, generator_model_name, init_prompt, f"eval_{eval_vlm}"
    )

    # Construct the full file path
    filename = f"{init_prompt}_numQ_{n_questions}_INIT.pkl"
    file_path = os.path.join(dir_path, filename)

    if not os.path.exists(file_path):
        logging.info(f"File {file_path} does not exist")
        return []

    # Load the image_score_pairs from the pickle file
    with open(file_path, "rb") as f:
        image_score_pairs = pickle.load(f)

    # Verify the structure of the loaded data
    assert isinstance(image_score_pairs, list), "Loaded data is not a list"
    for pair in image_score_pairs:
        assert (
            isinstance(pair, tuple) and len(pair) == 2
        ), "Each item must be a tuple of length 2"
        assert isinstance(
            pair[0], Image.Image
        ), "First element of each tuple must be a PIL Image"
        assert isinstance(
            pair[1], float
        ), "Second element of each tuple must be a float"

    return image_score_pairs


def save_softprompt_image_score_pairs(
    image_score_paris,
    output_dir,
    epoch_iter,
):
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    filename = f"image_score_pairs_epoch_{epoch_iter + 1}.pkl"
    with open(os.path.join(output_dir, filename), "wb") as f:
        pickle.dump(image_score_paris, f)


def load_softprompt_image_score_pairs(output_dir, epoch_iter):
    epoch_iter += 1
    # Construct the full file path
    filename = f"image_score_pairs_epoch_{epoch_iter}.pkl"
    file_path = os.path.join(output_dir, filename)
    if not os.path.exists(file_path):
        return []
    # Load the image_score_pairs from the pickle file
    with open(file_path, "rb") as f:
        image_score_pairs = pickle.load(f)

    # Verify the structure of the loaded data
    assert isinstance(image_score_pairs, list), "Loaded data is not a list"
    for pair in image_score_pairs:
        assert (
            isinstance(pair, tuple) and len(pair) == 2
        ), "Each item must be a tuple of length 2"
        assert isinstance(
            pair[0], Image.Image
        ), "First element of each tuple must be a PIL Image"
        assert isinstance(
            pair[1], float
        ), "Second element of each tuple must be a float"

    return image_score_pairs


def stable_diffusion_to_name(model_name):
    return model_name.split("/")[-1]


def resize_images_to_512(folder_path):
    parent_dir = os.path.dirname(folder_path)
    folder_name = os.path.basename(folder_path)
    new_folder_name = f"{folder_name}_512"
    new_folder_path = os.path.join(parent_dir, new_folder_name)
    os.makedirs(new_folder_path, exist_ok=True)
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
            image_path = os.path.join(folder_path, filename)
            try:
                with Image.open(image_path) as img:
                    resized_img = img.resize((512, 512), Image.Resampling.LANCZOS)
                    new_image_path = os.path.join(new_folder_path, filename)
                    resized_img.save(new_image_path, quality=95)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    print(f"\nResizing complete! Resized images saved in: {new_folder_path}")


def resize_image(img: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """
    Resize image to target size while maintaining aspect ratio and adding padding if necessary

    Args:
        img: PIL Image to resize
        target_size: Desired output size (width, height)

    Returns:
        Resized PIL Image
    """
    # Calculate aspect ratios
    target_ratio = target_size[0] / target_size[1]
    img_ratio = img.width / img.height

    # Create new image with white background
    new_img = Image.new("RGB", target_size, "white")

    if img_ratio > target_ratio:
        # Image is wider than target
        new_width = target_size[0]
        new_height = int(new_width / img_ratio)
        resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        # Center vertically
        y_offset = (target_size[1] - new_height) // 2
        new_img.paste(resized, (0, y_offset))
    else:
        # Image is taller than target
        new_height = target_size[1]
        new_width = int(new_height * img_ratio)
        resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        # Center horizontally
        x_offset = (target_size[0] - new_width) // 2
        new_img.paste(resized, (x_offset, 0))

    return new_img


def save_dict_to_csv(nested_dict, csv_file_path):
    fieldnames = ["image_name", "category", "question", "answer"]

    with open(csv_file_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for image_name, categories in nested_dict.items():
            for category, questions in categories.items():
                if not questions:
                    continue  # Skip empty categories
                for question, answer in questions.items():
                    writer.writerow(
                        {
                            "image_name": image_name,
                            "category": category,
                            "question": question,
                            "answer": answer,
                        }
                    )
