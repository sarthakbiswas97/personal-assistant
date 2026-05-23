FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# HF cache dir (writable on HF Spaces)
ENV HF_HOME=/tmp/hf_cache

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch (smaller image, works on both free and GPU tier)
# GPU tier on HF Spaces provides CUDA at runtime via nvidia drivers,
# but CPU torch still works there — and keeps the image small.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and evaluation results
COPY src/ src/
COPY eval/ eval/

# Download and cache the OSS model at build time
RUN python3 -c "\
from transformers import AutoModelForCausalLM, AutoTokenizer; \
import torch; \
model_name = 'Qwen/Qwen2.5-0.5B-Instruct'; \
AutoTokenizer.from_pretrained(model_name); \
AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)"

# HF Spaces expects port 7860
EXPOSE 7860

# Gradio config for container networking
ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

CMD ["python3", "-m", "src.app"]
