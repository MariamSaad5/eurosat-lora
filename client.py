"""Sends images to the running inference API and prints what it predicts.

Usage:
    python client.py samples/                 (every image in a folder)
    python client.py samples/Forest_1.jpg     (one image)
    python client.py samples/ --url http://localhost:8000
    python client.py monitoring_data/current --batch current --quiet

If a file name starts with a class name (e.g. Forest_1.jpg), the true
class is taken from it, sent to the API as the label (for performance
monitoring), and the script also reports how many were correct.
Use --no-labels to send predictions without labels, as in real traffic
where the true answer is not known yet.
"""
import argparse
import sys
from pathlib import Path

import requests

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="An image file or a folder of images")
    parser.add_argument("--url", default="http://localhost:8000", help="Where the API is running")
    parser.add_argument("--batch", help="Tag stored with every prediction, e.g. reference / current / drifted")
    parser.add_argument("--no-labels", action="store_true", help="Do not send true labels to the API")
    parser.add_argument("--quiet", action="store_true", help="Only print the summary, not one line per image")
    args = parser.parse_args()

    # 1. Is the service up?
    try:
        health = requests.get(f"{args.url}/health", timeout=10).json()
    except requests.ConnectionError:
        sys.exit(f"Could not reach {args.url}. Is the container running?")
    class_names = health["classes"]
    print(f"Service is up. Model: {health['model_path']} ({health['num_classes']} classes)\n")

    # 2. Collect the images
    path = Path(args.path)
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_TYPES) if path.is_dir() else [path]
    if not files:
        sys.exit(f"No images found in {path}")

    # 3. Send each one to /predict
    correct = labelled = 0
    if not args.quiet:
        print(f"{'file':<32}{'predicted':<22}{'confidence':>10}{'ms':>8}  result")
    for f in files:
        true_class = f.name.split("_")[0]
        form = {}
        if args.batch:
            form["batch"] = args.batch
        if true_class in class_names and not args.no_labels:
            form["label"] = true_class
        with open(f, "rb") as fh:
            response = requests.post(f"{args.url}/predict", files={"file": (f.name, fh)}, data=form, timeout=60)
        if response.status_code != 200:
            print(f"{f.name:<32}ERROR {response.status_code}: {response.json().get('detail')}")
            continue
        body = response.json()

        result = ""
        if true_class in class_names:
            labelled += 1
            correct += body["predicted_class"] == true_class
            result = "correct" if body["predicted_class"] == true_class else f"WRONG (true: {true_class})"
        if not args.quiet:
            print(f"{f.name:<32}{body['predicted_class']:<22}{body['confidence']:>10.4f}{body['inference_ms']:>8.1f}  {result}")

    print(f"\nSent {len(files)} images" + (f" (batch: {args.batch})" if args.batch else ""))
    if labelled:
        print(f"{correct}/{labelled} correct")


if __name__ == "__main__":
    main()