import base64
from openai import OpenAI
from PIL import Image
import io
import os
import torch


def encode_image_to_base64(image_path):
    """Convert image to base64 string"""
    if isinstance(image_path, str):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    elif isinstance(image_path, Image.Image):
        # Convert PIL Image to base64
        buffered = io.BytesIO()
        image_path.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    else:
        raise ValueError("Unsupported image type")


@torch.no_grad()
def get_chat_response_gpt(
    image,
    question,
    max_new_tokens=5,
    temperature=0,
):
    """
    Get response from GPT-4V for an image and question
    Args:
        image: Path to image or PIL Image
        question: Question to ask about the image
        max_new_tokens: Maximum number of tokens in response
    Returns:
        str: Response from GPT-4V
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    base64_image = encode_image_to_base64(image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        }
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        print("answer: :", response.choices[0].message.content)
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting response from GPT-4V: {str(e)}")
        return ""
