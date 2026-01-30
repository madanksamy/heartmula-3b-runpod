FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface

# Verify dependencies are installed correctly
RUN pip install --upgrade sentencepiece tiktoken && \
    python -c "import sentencepiece; import torch; print(f'PyTorch {torch.__version__}, SentencePiece OK')"

# Pre-download model (speeds up cold starts) - downloads config and tokenizer, model cached on first run
RUN python -c "from huggingface_hub import snapshot_download; \
    print('Pre-downloading HeartMuLa-3B model files...'); \
    snapshot_download('HeartMuLa/HeartMuLa-oss-3B', ignore_patterns=['*.bin', '*.pt'])"

CMD ["python", "-u", "handler.py"]
