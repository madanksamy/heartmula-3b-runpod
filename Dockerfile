FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

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

# Pre-download model (optional, speeds up cold starts)
# RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('amuvarma/HeartMuLa-3B', trust_remote_code=True); AutoModelForCausalLM.from_pretrained('amuvarma/HeartMuLa-3B', trust_remote_code=True)"

CMD ["python", "-u", "handler.py"]
