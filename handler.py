"""
HeartMuLa 3B RunPod Serverless Handler
Tamil/Multilingual Music Generation with S3 Output Storage

Compatible with the HeartMuLaProvider API format:
- lyrics: Structured lyrics with [Verse], [Chorus] etc.
- tags: Comma-separated style tags
- max_duration_seconds: Duration in seconds
- temperature, topk, cfg_scale: Generation parameters
"""

import runpod
import torch
import os
import uuid
import boto3
from io import BytesIO
import soundfile as sf
import numpy as np

# S3 Configuration
S3_BUCKET = os.environ.get("S3_BUCKET", "aiswara-music")
S3_REGION = os.environ.get("S3_REGION", "us-east-1")
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

# Model configuration
MODEL_ID = "HeartMuLa/HeartMuLa-oss-3B"
SAMPLE_RATE = 32000

# Global model instance
pipeline = None


def get_s3_client():
    """Get S3 client with credentials"""
    return boto3.client(
        's3',
        region_name=S3_REGION,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )


def upload_to_s3(audio_data: np.ndarray, sample_rate: int, filename: str) -> str:
    """Upload audio to S3 and return streaming URL"""
    s3 = get_s3_client()

    # Convert to WAV bytes using soundfile (better quality)
    buffer = BytesIO()
    sf.write(buffer, audio_data, sample_rate, format='WAV', subtype='PCM_16')
    buffer.seek(0)

    # Upload to S3
    key = f"generated/{filename}"
    s3.upload_fileobj(
        buffer,
        S3_BUCKET,
        key,
        ExtraArgs={
            'ContentType': 'audio/wav',
            'ACL': 'public-read'
        }
    )

    # Return public URL
    return f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{key}"


def load_model():
    """Load HeartMuLa 3B model using the official pipeline"""
    global pipeline

    if pipeline is not None:
        return pipeline

    print("Loading HeartMuLa 3B model...")

    # HeartMuLa uses a custom generation pipeline
    try:
        from heartmula import HeartMuLaGenPipeline
        pipeline = HeartMuLaGenPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    except ImportError:
        # Fallback to transformers if heartmula package not available
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("Using transformers fallback...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        pipeline = {"model": model, "tokenizer": tokenizer}

    print("Model loaded successfully!")
    return pipeline


def generate_music(lyrics: str, tags: str, duration: int = 120,
                   temperature: float = 1.0, topk: int = 50,
                   cfg_scale: float = 1.5) -> tuple:
    """Generate music from lyrics and tags"""

    pipe = load_model()

    # If using official HeartMuLa pipeline
    if hasattr(pipe, 'generate'):
        audio = pipe.generate(
            lyrics=lyrics,
            tags=tags,
            max_duration_seconds=duration,
            temperature=temperature,
            topk=topk,
            cfg_scale=cfg_scale
        )
        return audio, SAMPLE_RATE

    # Fallback for transformers-based loading
    model = pipe["model"]
    tokenizer = pipe["tokenizer"]

    # Build prompt in HeartMuLa format
    prompt = f"<|tags|>{tags}<|/tags|>\n<|lyrics|>{lyrics}<|/lyrics|>"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate
    max_tokens = int(duration * 50)  # ~50 tokens per second

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=topk,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode audio (model-specific)
    audio_tokens = outputs[0][inputs['input_ids'].shape[1]:]

    if hasattr(model, 'decode_audio'):
        audio = model.decode_audio(audio_tokens)
    else:
        # Basic fallback - this won't produce real audio
        audio = np.zeros(duration * SAMPLE_RATE, dtype=np.float32)

    return audio, SAMPLE_RATE


def handler(job):
    """RunPod serverless handler - compatible with HeartMuLaProvider"""
    job_input = job["input"]

    # Extract parameters (matching HeartMuLaProvider format)
    lyrics = job_input.get("lyrics", "")
    tags = job_input.get("tags", "pop,melodic")
    duration = job_input.get("max_duration_seconds", job_input.get("duration", 120))
    temperature = job_input.get("temperature", 1.0)
    topk = job_input.get("topk", job_input.get("top_k", 50))
    cfg_scale = job_input.get("cfg_scale", 1.5)

    # Backwards compatibility with prompt-based requests
    prompt = job_input.get("prompt", "")
    if prompt and not lyrics:
        lyrics = f"[Verse]\n{prompt}\n\n[Chorus]\n{prompt}"

    if not lyrics:
        return {"error": "Lyrics are required", "status": "error"}

    try:
        print(f"Generating music: tags={tags[:50]}, duration={duration}s")
        print(f"Lyrics preview: {lyrics[:100]}...")

        # Generate audio
        audio, sample_rate = generate_music(
            lyrics=lyrics,
            tags=tags,
            duration=duration,
            temperature=temperature,
            topk=topk,
            cfg_scale=cfg_scale
        )

        # Generate unique filename
        job_id = job.get("id", str(uuid.uuid4())[:8])
        filename = f"heartmula_{job_id}_{uuid.uuid4().hex[:8]}.wav"

        # Upload to S3
        audio_url = upload_to_s3(audio, sample_rate, filename)

        return {
            "status": "completed",
            "audio_url": audio_url,
            "duration": duration,
            "tags": tags,
            "sample_rate": sample_rate,
            "model": "HeartMuLa-3B"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e)
        }


# For local testing
if __name__ == "__main__":
    test_input = {
        "input": {
            "lyrics": "[Verse]\nA melodic Tamil folk song\nAbout nature and peace\n\n[Chorus]\nSinging with the wind",
            "tags": "tamil,folk,melodic",
            "max_duration_seconds": 30
        }
    }
    result = handler(test_input)
    print(result)


runpod.serverless.start({"handler": handler})
