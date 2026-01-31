FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install sentencepiece and tiktoken explicitly (needed for HeartMuLa tokenizer)
RUN pip install --no-cache-dir sentencepiece tiktoken

# Verify sentencepiece is working
RUN python -c "import sentencepiece; print('SentencePiece version:', sentencepiece.__version__)"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Pre-download model config and tokenizer files (speeds up cold starts)
RUN python -c "from huggingface_hub import snapshot_download; \
    print('Pre-downloading HeartMuLa-3B config files...'); \
    snapshot_download('HeartMuLa/HeartMuLa-oss-3B', ignore_patterns=['*.bin', '*.pt', '*.safetensors'])"

# Copy handler last (changes more frequently)
COPY handler.py .

CMD ["python", "-u", "handler.py"]
