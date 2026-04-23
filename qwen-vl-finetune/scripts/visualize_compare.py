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


def draw_bboxes(draw, bboxes, color, label_prefix="", thickness=3):
    for item in bboxes:
        x1, y1, x2, y2 = item["bbox"]
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        label = item["label"]
        for i in range(thickness):
            draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)
        text = f"{label_prefix}{label}"
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
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
    parser.add_argument('--nautilus_json', type=str, required=True)
    parser.add_argument('--baseline_json', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.nautilus_json, 'r') as f:
        nautilus_results = json.load(f)
    with open(args.baseline_json, 'r') as f:
        baseline_results = json.load(f)

    baseline_map = {item["image"]: item for item in baseline_results}

    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except (IOError, OSError):
        title_font = ImageFont.load_default()
    try:
        info_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except (IOError, OSError):
        info_font = ImageFont.load_default()

    gap = 15
    header_h = 55

    for idx, naut_item in enumerate(nautilus_results):
        image_name = naut_item["image"]
        base_item = baseline_map.get(image_name)
        if base_item is None:
            continue

        image_path = os.path.join(args.image_dir, image_name)
        img = Image.open(image_path).convert("RGB")
        ori_w, ori_h = img.size

        gt_bboxes = parse_bboxes(naut_item["ground_truth"])
        naut_bboxes = parse_bboxes(naut_item["prediction"])
        base_bboxes = parse_bboxes(base_item["prediction"])

        input_w, input_h = get_model_input_size(ori_w, ori_h)
        naut_bboxes = scale_pred_bboxes(naut_bboxes, ori_w, ori_h, input_w, input_h)
        base_bboxes = scale_pred_bboxes(base_bboxes, ori_w, ori_h, input_w, input_h)

        img_gt = img.copy()
        img_naut = img.copy()
        img_base = img.copy()

        draw_bboxes(ImageDraw.Draw(img_gt), gt_bboxes, color="#00FF00")
        draw_bboxes(ImageDraw.Draw(img_naut), naut_bboxes, color="#FF6B35")
        draw_bboxes(ImageDraw.Draw(img_base), base_bboxes, color="#FF0000")

        total_w = ori_w * 3 + gap * 2
        total_h = ori_h + header_h
        combined = Image.new("RGB", (total_w, total_h), (30, 30, 30))

        combined.paste(img_gt, (0, header_h))
        combined.paste(img_naut, (ori_w + gap, header_h))
        combined.paste(img_base, (ori_w * 2 + gap * 2, header_h))

        draw_c = ImageDraw.Draw(combined)

        gt_count = len(gt_bboxes)
        naut_count = len(naut_bboxes)
        base_count = len(base_bboxes)

        gt_title = f"Ground Truth ({gt_count})"
        naut_title = f"Nautilus ({naut_count})"
        base_title = f"Baseline ({base_count})"

        gt_tw = draw_c.textbbox((0, 0), gt_title, font=title_font)[2]
        naut_tw = draw_c.textbbox((0, 0), naut_title, font=title_font)[2]
        base_tw = draw_c.textbbox((0, 0), base_title, font=title_font)[2]

        draw_c.text(((ori_w - gt_tw) // 2, 8), gt_title, fill="#00FF00", font=title_font)
        draw_c.text((ori_w + gap + (ori_w - naut_tw) // 2, 8), naut_title, fill="#FF6B35", font=title_font)
        draw_c.text((ori_w * 2 + gap * 2 + (ori_w - base_tw) // 2, 8), base_title, fill="#FF0000", font=title_font)

        info = f"{image_name}"
        info_tw = draw_c.textbbox((0, 0), info, font=info_font)[2]
        draw_c.text(((total_w - info_tw) // 2, 35), info, fill="#AAAAAA", font=info_font)

        out_name = os.path.splitext(image_name)[0] + "_compare.png"
        combined.save(os.path.join(args.output_dir, out_name))
        print(f"[{idx+1}/{len(nautilus_results)}] {out_name}  GT:{gt_count} Nautilus:{naut_count} Baseline:{base_count}")

    print(f"\nAll saved to {args.output_dir}")


if __name__ == "__main__":
    main()
