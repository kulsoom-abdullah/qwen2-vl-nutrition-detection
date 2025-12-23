import re
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

SYSTEM_MESSAGE = (
    "You are a vision-language model specializing in nutrition-table detection.\n"
    "Detect every nutrition table in the image and respond only with lines of the form:\n"
    "nutrition-table<box(x_min, y_min),(x_max, y_max)>\n"
    "Coordinates are integers between 0 and 1000 in a normalized coordinate system (x first, then y).\n"
    "If multiple tables exist, return each on a separate line. Do not extract or describe text."
)

USER_PROMPT = "Detect all nutrition tables in this image and return the boxes."

def parse_bounding_boxes(response_text: str) -> list:
    """Parses model response text to extract normalized bounding box coordinates.

    Converts the Qwen2-VL specific integer coordinate format (0-1000) into
    normalized float coordinates [0.0, 1.0] required for standard IoU calculation.
    Handles variable spacing and ensures coordinate ordering (min < max).

    Args:
        response_text: Raw text output from the model (e.g., "nutrition-table<box(100,100),(200,200)>").

    Returns:
        A list of lists, where each inner list contains [x_min, y_min, x_max, y_max]
        as floats between 0.0 and 1.0. Returns empty list if no valid boxes found.
    """
    all_numbers_str = re.findall(r'[-+]?\d*\.\d+|\d+', response_text)
    if len(all_numbers_str) < 4:
        return []

    all_numbers = [float(n) for n in all_numbers_str]
    num_boxes = len(all_numbers) // 4

    parsed_boxes = []
    for i in range(num_boxes):
        start_index = i * 4
        box_nums = all_numbers[start_index : start_index + 4]
        c1, c2, c3, c4 = box_nums
        x1, y1, x2, y2 = c1 / 1000.0, c2 / 1000.0, c3 / 1000.0, c4 / 1000.0

        x_min = min(x1, x2)
        y_min = min(y1, y2)
        x_max = max(x1, x2)
        y_max = max(y1, y2)

        parsed_boxes.append([x_min, y_min, x_max, y_max])

    return parsed_boxes

def create_chat_format(sample: dict, downsize: bool = True, max_long_side: int = 1024) -> dict:
    """Formats a raw dataset sample into the Qwen2-VL chat conversation structure.

    Transforms OpenFoodFacts dataset entries into the multimodal chat format required
    by the Qwen2-VL processor. This includes normalizing bounding boxes to the 0-1000
    integer grid expected by the model's tokenizer.

    Args:
        sample: A dictionary containing 'image' (PIL.Image) and 'objects' (dict with 'bbox').
        downsize: Whether to resize the image before processing.
        max_long_side: Maximum dimension for the resized image.

    Returns:
        A dictionary with 'image' and 'messages' keys, ready for the processor.
    """
    assistant_response = ""
    objects = sample["objects"]

    if downsize:
        img = sample["image"].copy()
        img.thumbnail((max_long_side, max_long_side), Image.Resampling.LANCZOS)
        sample["image"] = img

    for i in range(len(objects["bbox"])):
        category = objects["category_name"][i]
        box = objects["bbox"][i]

        y_min_norm, x_min_norm, y_max_norm, x_max_norm = box

        x_min = int(x_min_norm * 1000)
        y_min = int(y_min_norm * 1000)
        x_max = int(x_max_norm * 1000)
        y_max = int(y_max_norm * 1000)

        assistant_response += (
            f"<|object_ref_start|>{category}<|object_ref_end|>"
            f"<|box_start|>({x_min},{y_min}),({x_max},{y_max})<|box_end|> "
        )

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample["image"]},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
        {"role": "assistant", "content": assistant_response.strip()},
    ]

    return {"image": sample["image"], "messages": messages}

class VLMDataCollator:
    """Collates and pads batches of multimodal data for Qwen2-VL training.

    Handles the complex requirement of masking user prompts in the loss calculation
    to ensure the model is only penalized for its generated responses (assistant turns),
    not the instructions it was given.

    Attributes:
        processor: The HuggingFace AutoProcessor for Qwen2-VL.
        mask_prompt: If True, sets label tokens corresponding to user prompts to -100 (ignored in loss).
    """

    def __init__(self, processor, mask_prompt: bool = True):
        self.processor = processor
        self.mask_prompt = mask_prompt
        self.pad_id = processor.tokenizer.pad_token_id

    def _to_multimodal_chat(self, conversation: list, image) -> list:
        """Helper to format the conversation list into the specific dict structure required by `apply_chat_template`."""
        formatted = []
        for message in conversation:
            role = message.get('role')
            content = message.get('content')

            if isinstance(content, list) and content and isinstance(content[0], dict) and 'type' in content[0]:
                formatted.append(message)
                continue

            text = content if isinstance(content, str) else ''
            if role == 'user':
                formatted.append({
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'image': image},
                        {'type': 'text', 'text': text.replace('<|image_1|>', '').strip()},
                    ],
                })
            else:
                formatted.append({
                    'role': role,
                    'content': [{'type': 'text', 'text': text}],
                })
        return formatted

    def __call__(self, features: list) -> dict:
        """Processes a list of samples into a batch tensor dictionary.

        Applies the chat template, processes vision info (patching), and pads text sequences.
        Crucially, it constructs the `labels` tensor by masking out user prompts if `mask_prompt` is True.
        """
        processed_conversations = []
        prompts = []
        image_inputs = []

        for feature in features:
            conversation = feature['messages']
            image = feature['image']

            multimodal = self._to_multimodal_chat(conversation, image)
            processed_conversations.append(multimodal)

            prompts.append(
                self.processor.apply_chat_template(
                    multimodal, tokenize=False, add_generation_prompt=False
                )
            )

            image_inputs.append(process_vision_info(multimodal)[0])

        batch = self.processor(
            text=prompts,
            images=image_inputs,
            return_tensors='pt',
            padding=True,
        )

        batch['pixel_values'] = batch['pixel_values'].to(torch.bfloat16)

        labels = batch['input_ids'].clone()
        for idx, conversation in enumerate(processed_conversations):
            prompt_only = conversation[:-1]
            if not prompt_only:
                continue
            prompt_text = self.processor.apply_chat_template(
                prompt_only, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = self.processor.tokenizer(
                prompt_text,
                add_special_tokens=False,
                return_attention_mask=False,
            ).input_ids
            if self.mask_prompt:
                labels[idx, : len(prompt_ids)] = -100

        if self.pad_id is not None:
            labels[batch['input_ids'] == self.pad_id] = -100

        batch['labels'] = labels
        return batch
