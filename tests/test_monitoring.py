import pytest

pytest.importorskip("fastapi")  # app/ needs the API requirements

from PIL import Image, ImageEnhance, ImageFilter

from app.monitoring import image_features


def _checkerboard():
    image = Image.new("RGB", (64, 64), (200, 180, 120))
    for x in range(0, 64, 8):
        for y in range(0, 64, 8):
            if (x + y) // 8 % 2:
                image.paste((40, 60, 30), (x, y, x + 8, y + 8))
    return image


def test_darker_image_has_lower_brightness():
    original = _checkerboard()
    darker = ImageEnhance.Brightness(original).enhance(0.6)
    assert image_features(darker)["brightness"] < image_features(original)["brightness"]


def test_blurred_image_has_lower_sharpness():
    original = _checkerboard()
    blurred = original.filter(ImageFilter.GaussianBlur(2))
    assert image_features(blurred)["sharpness"] < image_features(original)["sharpness"]


def test_colour_means_follow_the_image():
    features = image_features(Image.new("RGB", (8, 8), (255, 0, 0)))
    assert features["mean_red"] == 255 and features["mean_green"] == 0 and features["mean_blue"] == 0
