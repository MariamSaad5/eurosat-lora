"""Creates a drifted copy of a folder of images, to test drift monitoring.

Example:
    python make_drift.py monitoring_data/current monitoring_data/drifted

Every image is made darker and slightly blurred, as if the satellite
images now came from a different sensor, season or time of day. File
names are kept, so the true labels still match.
"""
import argparse
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Folder with the original images")
    parser.add_argument("target", help="Folder to write the drifted images to")
    parser.add_argument("--brightness", type=float, default=0.6, help="1.0 = unchanged, lower = darker")
    parser.add_argument("--blur", type=float, default=1.0, help="Gaussian blur radius in pixels, 0 = none")
    args = parser.parse_args()

    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(Path(args.source).glob("*.jpg")):
        image = Image.open(path).convert("RGB")
        image = ImageEnhance.Brightness(image).enhance(args.brightness)
        if args.blur > 0:
            image = image.filter(ImageFilter.GaussianBlur(args.blur))
        image.save(target / path.name, quality=95)
        count += 1
    print(f"Wrote {count} drifted images to {target} (brightness x{args.brightness}, blur radius {args.blur})")


if __name__ == "__main__":
    main()
