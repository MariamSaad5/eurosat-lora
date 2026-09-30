# Small official Linux image with Python 3.11 already installed
FROM python:3.11-slim

# Which PyTorch build to install. "cpu" by default.
# On a machine with an NVIDIA GPU, build with: --build-arg TORCH_VARIANT=cu130
ARG TORCH_VARIANT=cpu

# PYTHONUNBUFFERED: print logs immediately instead of holding them in a buffer
# PIP_NO_CACHE_DIR: don't keep pip's download cache inside the image
# HF_HOME: where Hugging Face saves the ViT weights (a volume can be attached here)
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface

# All following commands run from /app
WORKDIR /app

# Install PyTorch first, from PyTorch's own download site
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/${TORCH_VARIANT}

# Install the other libraries. Copied before the code so Docker can reuse
# this cached step when only .py files change.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the project code into the image
COPY main.py .
COPY src/ src/
COPY configs/ configs/

# The container always runs main.py. The config below is only the default
# and is replaced by anything you add at the end of "docker run".
ENTRYPOINT ["python", "main.py"]
CMD ["--config", "configs/docker_sanity.yaml"]
