"""Observability helpers for the inference service (Assignment 16).

Two kinds of monitoring live here:

1. Prediction logging, for data drift and model performance.
   Every request appends one row to a CSV file: simple statistics that
   describe the input image, what the model predicted, how confident it
   was, and the true label if the caller knows it. monitor.py later runs
   Evidently on this file.

2. Prometheus metrics, for live and infrastructure monitoring.
   Counters and histograms that Prometheus scrapes from /metrics every few
   seconds and Grafana turns into dashboards.
"""
import csv
import os
import threading
from datetime import datetime, timezone

import numpy as np
from PIL import Image
from prometheus_client import Counter, Histogram

# ---------- 1. image statistics ----------

FEATURE_COLUMNS = ["brightness", "contrast", "sharpness", "mean_red", "mean_green", "mean_blue"]


def image_features(image: Image.Image) -> dict:
    """A handful of numbers that summarize what an image looks like.

    The model sees pixels, but drift tests need a small table of numbers,
    so each image is described by:
      brightness  average grey level (0-255)
      contrast    spread of grey levels (standard deviation)
      sharpness   average difference between neighbouring pixels; drops
                  when an image is blurred
      mean_red / mean_green / mean_blue   average of each colour channel
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    grey = rgb.mean(axis=2)
    sharpness = (np.abs(np.diff(grey, axis=0)).mean() + np.abs(np.diff(grey, axis=1)).mean()) / 2
    return {
        "brightness": round(float(grey.mean()), 3),
        "contrast": round(float(grey.std()), 3),
        "sharpness": round(float(sharpness), 3),
        "mean_red": round(float(rgb[..., 0].mean()), 3),
        "mean_green": round(float(rgb[..., 1].mean()), 3),
        "mean_blue": round(float(rgb[..., 2].mean()), 3),
    }


# ---------- 2. prediction log ----------

LOG_COLUMNS = ["timestamp", "batch", "filename", *FEATURE_COLUMNS,
               "prediction", "confidence", "label", "inference_ms"]

_log_lock = threading.Lock()  # two requests at once must not write over each other


def log_prediction(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    row = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), **row}
    with _log_lock:
        new_file = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            if new_file:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in LOG_COLUMNS})


# ---------- 3. Prometheus metrics ----------

PREDICTIONS = Counter(
    "model_predictions_total", "Predictions made, by predicted class", ["predicted_class"]
)
CONFIDENCE = Histogram(
    "model_prediction_confidence", "Confidence of the top prediction",
    buckets=[0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
)
INFERENCE_SECONDS = Histogram(
    "model_inference_seconds", "Time spent inside the model only",
    buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0],
)
LABELLED = Counter(
    "model_labelled_predictions_total",
    "Predictions that came with a true label, split by whether they were correct",
    ["correct"],
)
