import json
import argparse
import os
import re
from PIL import Image, ImageDraw, ImageFont


def parse_bboxes(text):
    pattern = r'\{\s*"bbox_2d"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\s*,\s*"label"\s*:\s*"([^"]+)"\s*\}'
    results = []
    for m in re.finditer(pattern, text):
        bbox = [int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))]
        label = m.group(5)
        results.append({"bbox": bbox, "label": label})
    return results


def scale_pred_bboxes(bboxes, ori_w, ori_h, input_w, input_h):
    scaled = []
    w_scale = ori_w / input_w
    h_scale = ori_h / input_h
    for item in bboxes:
        x1, y1, x2, y2 = item["bbox"]
        scaled.append({
            "bbox": [round(x1 * w_scale), round(y1 * h_scale), round(x2 * w_scale), round(y2 * h_scale)],
            "label": item["label"],
        })
    return scaled


def get_model_input_size(ori_w, ori_h, min_pixels=784, max_pixels=1048992):
    ratio = ori_w / ori_h
    target_pixels = min(max(ori_w * ori_h, min_pixels), max_pixels)
    h = int((target_pixels / ratio) ** 0.5)
    w = int(h * ratio)
    h = (h // 28) * 28
    w = (w // 28) * 28
    h = max(h, 28)
    w = max(w, 28)
    total = h * w
    if total > max_pixels:
        scale = (max_pixels / total) ** 0.5
        h = int(h * scale) // 28 * 28
        w = int(w * scale) // 28 * 28
    return w, h


LABEL_COLORS = {
    "massive": "#FF6B6B",
    "encrusting": "#4ECDC4",
    "branching": "#45B7D1",
    "laminar": "#96CEB4",
    "folliaceous": "#FFEAA7",
    "columnar": "#DDA0DD",
}


def draw_bboxes(draw, bboxes, color, label_prefix="", thickness=3):
    for item in bboxes:
        x1, y1, x2, y2 = item["bbox"]
        label = item["label"]
        for i in range(thickness):
            draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)
        text = f"{label_prefix}{label}"
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except (IOError, OSError):
            font = ImageFont.load_default()
        bbox_text = draw.textbbox((0, 0), text, font=font)
        tw = bbox_text[2] - bbox_text[0]
        th = bbox_text[3] - bbox_text[1]
        ty = max(y1 - th - 4, 0)
        draw.rectangle([x1, ty, x1 + tw + 4, ty + th + 4], fill=color)
        draw.text((x1 + 2, ty + 2), text, fill="white", font=font)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_json', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.results_json, 'r') as f:
        results = json.load(f)

    for idx, item in enumerate(results):
        image_path = os.path.join(args.image_dir, item["image"])
        img = Image.open(image_path).convert("RGB")
        ori_w, ori_h = img.size

        gt_bboxes = parse_bboxes(item["ground_truth"])
        pred_bboxes = parse_bboxes(item["prediction"])

        input_w, input_h = get_model_input_size(ori_w, ori_h)
        pred_bboxes = scale_pred_bboxes(pred_bboxes, ori_w, ori_h, input_w, input_h)

        img_gt = img.copy()
        img_pred = img.copy()

        draw_gt = ImageDraw.Draw(img_gt)
        draw_pred = ImageDraw.Draw(img_pred)

        draw_bboxes(draw_gt, gt_bboxes, color="#00FF00")
        draw_bboxes(draw_pred, pred_bboxes, color="#FF0000")

        combined = Image.new("RGB", (ori_w * 2 + 20, ori_h + 60), (40, 40, 40))
        combined.paste(img_gt, (0, 60))
        combined.paste(img_pred, (ori_w + 20, 60))

        draw_combined = ImageDraw.Draw(combined)
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except (IOError, OSError):
            title_font = ImageFont.load_default()

        draw_combined.text((ori_w // 2 - 80, 15), "Ground Truth", fill="#00FF00", font=title_font)
        draw_combined.text((ori_w + 20 + ori_w // 2 - 60, 15), "Prediction", fill="#FF0000", font=title_font)

        gt_count = len(gt_bboxes)
        pred_count = len(pred_bboxes)
        info = f"GT: {gt_count} objects | Pred: {pred_count} objects"
        try:
            info_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except (IOError, OSError):
            info_font = ImageFont.load_default()
        info_w = draw_combined.textbbox((0, 0), info, font=info_font)[2]
        draw_combined.text(((ori_w * 2 + 20 - info_w) // 2, 40), info, fill="white", font=info_font)

        out_name = os.path.splitext(item["image"])[0] + "_vis.png"
        combined.save(os.path.join(args.output_dir, out_name))
        print(f"[{idx+1}/{len(results)}] {out_name} saved (GT: {gt_count}, Pred: {pred_count})")

    print(f"\nAll visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()
