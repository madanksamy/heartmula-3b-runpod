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

# Pre-download model (speeds up cold starts significantly)
RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
    print('Downloading HeartMuLa-3B tokenizer...'); \
    AutoTokenizer.from_pretrained('HeartMuLa/HeartMuLa-oss-3B', trust_remote_code=True); \
    print('Downloading HeartMuLa-3B model (~6GB)...'); \
    AutoModelForCausalLM.from_pretrained('HeartMuLa/HeartMuLa-oss-3B', trust_remote_code=True)"

CMD ["python", "-u", "handler.py"]
