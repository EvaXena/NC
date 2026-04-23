import json
import argparse
import torch
import sys
import os
import re
from pathlib import Path
from typing import List
from PIL import Image

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from transformers import AutoProcessor
from qwenvl.nautilus_model.Qwen2_5_VL_Nautilus_ForConditionalGeneration import Qwen2_5_VL_Nautilus_ForConditionalGeneration
from qwen_vl_utils import process_vision_info

image_token_id = 151655


def scale_bboxes_in_text(text: str, w_scale: float, h_scale: float) -> str:
    def scale_bbox(match):
        x1 = int(match.group(1))
        y1 = int(match.group(2))
        x2 = int(match.group(3))
        y2 = int(match.group(4))
        new_x1 = round(x1 * w_scale)
        new_y1 = round(y1 * h_scale)
        new_x2 = round(x2 * w_scale)
        new_y2 = round(y2 * h_scale)
        return f"[{new_x1}, {new_y1}, {new_x2}, {new_y2}]"

    pattern = r"\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\]"
    scaled_text = re.sub(pattern, scale_bbox, text)
    return scaled_text


def get_grid_thw(processor, image_file):
    image = Image.open(image_file).convert("RGB")
    width, height = image.size
    visual_processed = processor.preprocess(image, return_tensors="pt")
    image_tensor = visual_processed["pixel_values"]
    if isinstance(image_tensor, List):
        image_tensor = image_tensor[0]
    grid_thw = visual_processed["image_grid_thw"][0]
    return grid_thw, width, height


def double_image_tokens(inputs: dict, token_id: int) -> torch.Tensor:
    input_ids = inputs['input_ids'].squeeze(0)
    attention_mask = inputs['attention_mask'].squeeze(0)
    new_ids, new_mask = [], []

    for token, mask in zip(input_ids, attention_mask):
        new_ids.append(token.item())
        new_mask.append(mask.item())
        if token.item() == token_id:
            new_ids.append(token.item())
            new_mask.append(mask.item())

    return (
        torch.tensor(new_ids, dtype=input_ids.dtype, device=input_ids.device).unsqueeze(0),
        torch.tensor(new_mask, dtype=attention_mask.dtype, device=attention_mask.device).unsqueeze(0),
    )


def run_inference(model, processor, image_path, prompt):
    image_processor = processor.image_processor
    grid_thw, ori_w, ori_h = get_grid_thw(image_processor, image_path)
    input_height = grid_thw[1].item() * 14
    input_width = grid_thw[2].item() * 14
    scale_h, scale_w = input_height / ori_h, input_width / ori_w
    prompt_scaled = scale_bboxes_in_text(prompt, scale_w, scale_h)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_scaled},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs['input_ids'], inputs['attention_mask'] = double_image_tokens(inputs, image_token_id)
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=2048)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    res_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return res_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to merged model directory')
    parser.add_argument('--test_json', type=str, required=True, help='Path to test_nautilus.json')
    parser.add_argument('--image_dir', type=str, required=True, help='Path to test images directory')
    parser.add_argument('--output', type=str, required=True, help='Path to save results JSON')
    parser.add_argument('--max_samples', type=int, default=0, help='Max number of samples to run (0 = all)')
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    model = Qwen2_5_VL_Nautilus_ForConditionalGeneration.from_pretrained(
        args.checkpoint,
        cache_dir=None,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()

    min_pixels = 1 * 28 * 28
    max_pixels = 1338 * 28 * 28
    processor = AutoProcessor.from_pretrained(args.checkpoint, min_pixels=min_pixels, max_pixels=max_pixels)

    with open(args.test_json, 'r') as f:
        test_data = json.load(f)

    total = len(test_data)
    if args.max_samples > 0:
        test_data = test_data[:args.max_samples]
        total = len(test_data)
    print(f"Total test samples: {total}")

    results = []
    for idx, sample in enumerate(test_data):
        image_name = sample["image"]
        image_path = os.path.join(args.image_dir, image_name)

        prompt = sample["conversations"][0]["value"]
        prompt = prompt.replace("<image>\n", "").replace("<image>", "")

        ground_truth = sample["conversations"][1]["value"]

        prediction = run_inference(model, processor, image_path, prompt)

        results.append({
            "id": sample["id"],
            "image": image_name,
            "prediction": prediction,
            "ground_truth": ground_truth,
        })

        print(f"[{idx+1}/{total}] {image_name} done")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {args.output}")
    print(f"Total predictions: {len(results)}")


if __name__ == "__main__":
    main()
