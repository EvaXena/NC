import json
import argparse
import os

def convert(input_path, output_path, split):
    with open(input_path, "r") as f:
        data = json.load(f)

    converted = []
    for item in data:
        new_item = {
            "id": [item["id"], "5"],
            "image": item["image"].replace(f"{split}/", "", 1),
            "conversations": item["conversations"],
        }
        converted.append(new_item)

    with open(output_path, "w") as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(converted)} samples: {input_path} -> {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    if args.input and args.output:
        convert(args.input, args.output, args.split)
    else:
        convert(
            os.path.join(base_dir, "train.json"),
            os.path.join(base_dir, "train_nautilus.json"),
            "train",
        )
        convert(
            os.path.join(base_dir, "test.json"),
            os.path.join(base_dir, "test_nautilus.json"),
            "test",
        )
