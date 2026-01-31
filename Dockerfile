FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install heartlib from GitHub (HeartMuLa official library)
RUN pip install git+https://github.com/HeartMuLa/heartlib.git

# Install additional dependencies needed by heartlib
RUN pip install --no-cache-dir sentencepiece tiktoken

# Create checkpoint directory
RUN mkdir -p /app/ckpt

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface
ENV HEARTMULA_MODEL_PATH=/app/ckpt

# Pre-download the HeartMuLa models and config files (without hf_transfer for compatibility)
# Structure: /app/ckpt/tokenizer.json, gen_config.json, HeartMuLa-oss-3B/, HeartCodec-oss/
RUN pip install huggingface_hub[cli] && \
    HF_HUB_ENABLE_HF_TRANSFER=0 huggingface-cli download --local-dir /app/ckpt HeartMuLa/HeartMuLaGen && \
    HF_HUB_ENABLE_HF_TRANSFER=0 huggingface-cli download --local-dir /app/ckpt/HeartMuLa-oss-3B HeartMuLa/HeartMuLa-RL-oss-3B-20260123 && \
    HF_HUB_ENABLE_HF_TRANSFER=0 huggingface-cli download --local-dir /app/ckpt/HeartCodec-oss HeartMuLa/HeartCodec-oss-20260123

# Copy handler last (changes more frequently)
COPY handler.py .

CMD ["python", "-u", "handler.py"]
