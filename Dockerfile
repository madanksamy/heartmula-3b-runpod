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
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HEARTMULA_MODEL_PATH=/app/ckpt

# Pre-download the HeartMuLa and HeartCodec models
RUN pip install huggingface_hub[cli] && \
    huggingface-cli download --local-dir /app/ckpt/HeartMuLa-oss-3B HeartMuLa/HeartMuLa-RL-oss-3B-20260123 && \
    huggingface-cli download --local-dir /app/ckpt/HeartCodec-oss HeartMuLa/HeartCodec-oss-20260123

# Copy handler last (changes more frequently)
COPY handler.py .

CMD ["python", "-u", "handler.py"]
