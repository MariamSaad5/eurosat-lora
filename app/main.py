"""FastAPI inference service for the INT8 EuroSAT ViT.

Run locally:   uvicorn app.main:app --port 8000
Endpoints:
    GET  /health    is the service up, and which model is loaded
    POST /predict   upload one image, get the land-use class back
    GET  /metrics   Prometheus metrics (requests, latency, predictions, CPU, memory)
"""
import io
import os
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from prometheus_fastapi_instrumentator import Instrumentator

from app import monitoring
from src.quantize import load_quantized

MODEL_PATH = os.getenv("MODEL_PATH", "outputs/vit_lora_r16_int8.pt")
PREDICTION_LOG = os.getenv("PREDICTION_LOG", "logs/predictions.csv")
TOP_K = int(os.getenv("TOP_K", "3"))

# Filled in once at startup, then shared by every request.
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once when the server starts, not on every request:
    # loading takes seconds, a prediction takes a fraction of a second.
    model, processor, class_names = load_quantized(MODEL_PATH)
    state.update(model=model, processor=processor, class_names=class_names)
    yield
    state.clear()


app = FastAPI(title="EuroSAT ViT (LoRA + INT8)", lifespan=lifespan)

# Adds a /metrics endpoint with request counts and response times for every
# endpoint, plus the process CPU and memory metrics prometheus_client collects.
Instrumentator().instrument(app).expose(app, include_in_schema=False)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "num_classes": len(state["class_names"]),
        "classes": state["class_names"],
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    label: str | None = Form(None),   # true class, if the caller knows it
    batch: str | None = Form(None),   # optional tag to group requests, e.g. "reference"
):
    # 1. Read the upload and open it as an image; reject anything that isn't one
    data = await file.read()
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable image.")

    # 2. Preprocess exactly like training: resize to 224x224 and normalize
    pixel_values = state["processor"](image, return_tensors="pt")["pixel_values"]

    # 3. Run the INT8 model
    start = time.perf_counter()
    with torch.inference_mode():
        logits = state["model"](pixel_values=pixel_values).logits
    inference_ms = (time.perf_counter() - start) * 1000

    # 4. Turn scores into probabilities and return the top classes
    probs = torch.softmax(logits, dim=-1)[0]
    scores, indices = probs.topk(min(TOP_K, probs.numel()))
    top = [
        {"class": state["class_names"][i], "confidence": round(s, 4)}
        for s, i in zip(scores.tolist(), indices.tolist())
    ]

    # 5. Monitoring: Prometheus metrics and one row in the prediction log
    predicted, confidence = top[0]["class"], top[0]["confidence"]
    monitoring.PREDICTIONS.labels(predicted_class=predicted).inc()
    monitoring.CONFIDENCE.observe(confidence)
    monitoring.INFERENCE_SECONDS.observe(inference_ms / 1000)
    if label in state["class_names"]:
        monitoring.LABELLED.labels(correct=str(label == predicted).lower()).inc()
    else:
        label = None  # unknown or missing labels are not used for accuracy
    monitoring.log_prediction(PREDICTION_LOG, {
        "batch": batch or "",
        "filename": file.filename,
        **monitoring.image_features(image),
        "prediction": predicted,
        "confidence": confidence,
        "label": label or "",
        "inference_ms": round(inference_ms, 1),
    })

    return {
        "filename": file.filename,
        "predicted_class": top[0]["class"],
        "confidence": top[0]["confidence"],
        "top_k": top,
        "inference_ms": round(inference_ms, 1),
    }