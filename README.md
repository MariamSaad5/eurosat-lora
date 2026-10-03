# LoRA Fine-Tuning for EuroSAT (ViT)

A manual (from scratch, no PEFT library) LoRA fine-tuning pipeline that adapts
`google/vit-base-patch16-224` to EuroSAT RGB satellite image classification.
The best adapter is merged into the ViT, quantized to INT8, and served by a
FastAPI inference service running in Docker. The project is tested
automatically with GitHub Actions.

## Repository structure

```
eurosat-lora/
├── main.py                  entry point: runs one experiment from a config
├── quantize.py              merges the best LoRA checkpoint and quantizes it to INT8
├── client.py                sends images to the running API and prints predictions
├── app/main.py              FastAPI inference service (/health, /predict)
├── samples/                 20 EuroSAT test images (2 per class) for trying the API
├── src/
│   ├── config.py            YAML config into ExperimentConfig / LoRAConfig
│   ├── data.py              dataset loading, 70/15/15 split, DataLoaders
│   ├── lora.py              LoRALinear layer, apply_lora_to_vit, merge_lora_weights
│   ├── quantize.py          load checkpoint, INT8 quantization, save/load INT8 model
│   ├── metrics.py           parameter counts and checkpoint sizes
│   ├── train.py             shared train/eval epoch loop
│   ├── finetuner.py         BaseFineTuner and LoRAFineTuner
│   └── utils.py             seeding
├── configs/                 one YAML per experiment (Kaggle and Docker versions)
├── tests/                   pytest tests (no dataset or model download needed)
├── kaggle/kaggle_run.ipynb  notebook that runs the real experiments on Kaggle
├── Dockerfile               training image
├── Dockerfile.api           inference image (FastAPI + INT8 model)
└── .github/workflows/ci.yml the CI pipeline
```

## Design

`BaseFineTuner` owns everything that is the same for every fine-tuning
strategy: the optimizer, the epoch loop, timing, GPU memory tracking and
checkpoint saving. `LoRAFineTuner` inherits all of that and only overrides
`build_model()`, `get_checkpoint_state_dict()` and `extra_result_fields()`.

## Branches

- `main`: stable code. Every push is tested, built and published as a Docker image.
- `develop`: work in progress. Every push is tested and built, but not published.

## CI pipeline (GitHub Actions)

| Job | What it does | Runs on |
|---|---|---|
| `test` | installs CPU PyTorch and runs `pytest` | pushes to `main` and `develop`, pull requests to `main` |
| `build` | builds the training image and runs the tests inside it, then builds the inference API image | same as `test`, only if `test` passed |
| `deploy` | builds and pushes the training image to GitHub Container Registry (ghcr.io) | pushes to `main` only, only if `build` passed |

## Running the tests locally

```
pip install -r requirements-dev.txt
pytest -v
```

## Training on Kaggle

Open `kaggle/kaggle_run.ipynb` in Kaggle (File, Import Notebook), attach the
EuroSAT RGB dataset, turn on the GPU and Internet, and run all cells. The
configs in `configs/lora_r*.yaml` already use Kaggle paths.

## Running with Docker

```
docker build -t eurosat-lora:cpu .

docker run --name eurosat-sanity --shm-size=2g \
  -v /path/to/EuroSAT_RGB:/data/EuroSAT_RGB \
  -v /path/to/outputs:/outputs \
  -v hf_cache:/cache/huggingface \
  eurosat-lora:cpu
```

Add `--config configs/docker_lora_r16.yaml` at the end to run the full
experiment instead of the sanity check. On a machine with an NVIDIA GPU, build
with `--build-arg TORCH_VARIANT=cu130` and add `--gpus all` to `docker run`.

## Assignment 14: optimized inference pipeline

```
train (Kaggle GPU)  ->  lora_r16_best.pt  ->  quantize.py  ->  vit_lora_r16_int8.pt  ->  FastAPI in Docker
```

1. **Best LoRA model.** Training saves `<name>_best.pt` from the epoch with the
   lowest validation loss. The file holds only the LoRA and classifier weights
   (0.6 MB) plus everything needed to rebuild the model.
2. **Quantization.** `quantize.py` merges LoRA into the base weights
   (`W = W_base + (alpha / r) * B @ A`) and applies dynamic INT8 quantization
   to every Linear layer. It checks that the merge changes nothing and that the
   saved file reloads to identical outputs.
3. **Inference service.** `app/main.py` loads the INT8 model once at startup
   and serves `GET /health` and `POST /predict` (upload an image, get the class,
   confidence and top 3 back).
4. **Docker.** `Dockerfile.api` builds a CPU-only image with just the service
   code. The model file is mounted at run time, not copied into the image.

### Running it

```
python quantize.py --checkpoint outputs/lora_r16_best.pt --output outputs/vit_lora_r16_int8.pt

docker build -f Dockerfile.api -t eurosat-api:cpu .
docker run -d --name eurosat-api -p 8000:8000 -v "${PWD}/outputs:/models:ro" eurosat-api:cpu
python client.py samples
```

Interactive API docs: http://localhost:8000/docs

### Results (lora_r16)

| | FP32 (merged) | INT8 (dynamic) |
|---|---|---|
| Model size | 327.4 MB | 84.4 MB (3.88x smaller) |
| Test accuracy, first 500 test images, CPU | 96.80% | 96.40% |
| CPU latency, batches of 32 (Kaggle) | 169.3 ms/image | 146.4 ms/image |

Best checkpoint: epoch 1 of 3 (val loss 0.1089, val accuracy 96.57%).
Through the Dockerized API, 19 of the 20 sample images were classified
correctly, at about 250 ms per single-image request on a laptop (Docker Desktop).

### Limitations

- Only Linear layers are quantized; attention scores, softmax, LayerNorm and
  GELU stay FP32, so the speed-up is much smaller than the size reduction.
- `torch.ao.quantization` is deprecated in favour of torchao. It still works in
  PyTorch 2.14.1, which is why that version is pinned in `Dockerfile.api`.
- The model always picks one of the 10 EuroSAT classes, even for images that
  are not satellite tiles. Low confidence is the only warning sign.

## Training results

Each run appends one row to `<output_dir>/results.csv` with trainable
parameters, validation accuracy and F1, training time, GPU memory, and the
size of the adapter-only checkpoint.

   Edited on GitHub.