import io
import base64
import json
import numpy as np

from google import genai
from google.genai import types
from PIL import Image
from loguru import logger

from vlm_recog.models import DetectedItem, DetectedItems


def parse_json(json_output: str) -> str:
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return output


def detect(image: Image.Image, labels: list[str]) -> DetectedItems:
    if not labels:
        raise ValueError("labels is empty")

    client = genai.Client()
    image.thumbnail([1024, 1024], Image.Resampling.LANCZOS)
    labels_tokens = ", ".join([f'"{l}"' for l in labels[:-1]])
    if len(labels) == 1:
        labels_tokens += labels[0]
    else:
        labels_tokens += f" and \"{labels[-1]}\""

    prompt = f"""
    Give the segmentation masks for the {labels_tokens}.
    Output a JSON list of segmentation masks where each entry contains the 2D
    bounding box in the key "box_2d", the segmentation mask in key "mask", and
    the text label in the key "label". Use descriptive labels.
    """
    logger.debug(f"Prompt: {prompt}")

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt, image],
        config=config
    )

    # Parse JSON response
    logger.debug(f"Response: {response.text}")
    items = json.loads(parse_json(response.text))

    detected_items: DetectedItems = []
    # Process each mask
    for item in items:
        # Get bounding box coordinates
        box = item["box_2d"]
        y0 = int(box[0] / 1000 * image.size[1])
        x0 = int(box[1] / 1000 * image.size[0])
        y1 = int(box[2] / 1000 * image.size[1])
        x1 = int(box[3] / 1000 * image.size[0])

        # Skip invalid boxes
        if y0 >= y1 or x0 >= x1:
            continue

        # Process mask
        png_str = item["mask"]
        if not png_str.startswith("data:image/png;base64,"):
            continue

        # Remove prefix
        png_str = png_str.removeprefix("data:image/png;base64,")
        mask_data = base64.b64decode(png_str)
        mask = Image.open(io.BytesIO(mask_data))

        # Resize mask to match bounding box
        mask = mask.resize((x1 - x0, y1 - y0), Image.Resampling.BILINEAR)

        # Convert mask to numpy array for processing
        mask_array = np.array(mask)

        # Create segmentation mask
        detected_item = DetectedItem(
            box_2d=(x0, y0, x1, y1),
            segmentation_mask=mask_array,
            label=item["label"]
        )
        detected_items.append(detected_item)

    return detected_items
