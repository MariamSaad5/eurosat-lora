# LoRA Fine-Tuning for EuroSAT (ViT)

A manual (from scratch, no PEFT library) LoRA fine-tuning pipeline that adapts
`google/vit-base-patch16-224` to EuroSAT RGB satellite image classification.
The project is packaged with Docker and tested automatically with GitHub Actions.

## Repository structure

```
eurosat-lora/
├── main.py                  entry point: runs one experiment from a config
├── src/
│   ├── config.py            YAML config into ExperimentConfig / LoRAConfig
│   ├── data.py              dataset loading, 70/15/15 split, DataLoaders
│   ├── lora.py              LoRALinear layer and apply_lora_to_vit
│   ├── metrics.py           parameter counts and checkpoint sizes
│   ├── train.py             shared train/eval epoch loop
│   ├── finetuner.py         BaseFineTuner and LoRAFineTuner
│   └── utils.py             seeding
├── configs/                 one YAML per experiment (Kaggle and Docker versions)
├── tests/                   pytest tests (no dataset or model download needed)
├── kaggle/kaggle_run.ipynb  notebook that runs the real experiments on Kaggle
├── Dockerfile               builds the project image
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
| `build` | builds the Docker image and runs the tests inside a container | same as `test`, only if `test` passed |
| `deploy` | builds and pushes the image to GitHub Container Registry (ghcr.io) | pushes to `main` only, only if `build` passed |

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

## Results

Each run appends one row to `<output_dir>/results.csv` with trainable
parameters, validation accuracy and F1, training time, GPU memory, and the
size of the adapter-only checkpoint.

   Edited on GitHub.
