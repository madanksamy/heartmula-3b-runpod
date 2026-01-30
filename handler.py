"""
HeartMuLa 3B RunPod Serverless Handler
Tamil/Multilingual Music Generation with S3 Output Storage
"""

import runpod
import torch
import os
import uuid
import boto3
from io import BytesIO
import scipy.io.wavfile as wavfile
import numpy as np

# S3 Configuration
S3_BUCKET = os.environ.get("S3_BUCKET", "aiswara-music")
S3_REGION = os.environ.get("S3_REGION", "us-east-1")
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

# Model configuration
MODEL_ID = "amuvarma/HeartMuLa-3B"
SAMPLE_RATE = 32000

# Global model instance
model = None
tokenizer = None


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

    # Convert to WAV bytes
    buffer = BytesIO()
    wavfile.write(buffer, sample_rate, audio_data.astype(np.int16))
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
    """Load HeartMuLa 3B model"""
    global model, tokenizer

    if model is not None:
        return model, tokenizer

    print("Loading HeartMuLa 3B model...")

    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    print("Model loaded successfully!")
    return model, tokenizer


def generate_music(prompt: str, duration: int = 30, temperature: float = 0.8,
                   language: str = "tamil", instrumental: bool = False) -> dict:
    """Generate music from text prompt"""

    model, tokenizer = load_model()

    # Build prompt based on language/style
    if instrumental:
        full_prompt = f"[INST] Generate instrumental music: {prompt} [/INST]"
    elif language == "tamil":
        full_prompt = f"[INST] Generate Tamil song: {prompt} [/INST]"
    else:
        full_prompt = f"[INST] Generate music: {prompt} [/INST]"

    # Tokenize
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    # Generate
    max_tokens = int(duration * 50)  # ~50 tokens per second of audio

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode audio from tokens
    audio_tokens = outputs[0][inputs['input_ids'].shape[1]:]

    # Convert to audio (model-specific decoding)
    if hasattr(model, 'decode_audio'):
        audio = model.decode_audio(audio_tokens)
    else:
        # Fallback: treat tokens as audio codes
        audio = audio_tokens.cpu().numpy().astype(np.float32)
        audio = (audio / audio.max() * 32767).astype(np.int16)

    return audio


def handler(job):
    """RunPod serverless handler"""
    job_input = job["input"]

    # Extract parameters
    prompt = job_input.get("prompt", "")
    duration = job_input.get("duration", 30)
    temperature = job_input.get("temperature", 0.8)
    language = job_input.get("language", "tamil")
    instrumental = job_input.get("instrumental", False)
    lyrics = job_input.get("lyrics", "")

    if not prompt and not lyrics:
        return {"error": "Either prompt or lyrics is required"}

    # Use lyrics as prompt if provided
    if lyrics:
        prompt = f"Lyrics: {lyrics}\nStyle: {prompt}" if prompt else f"Lyrics: {lyrics}"

    try:
        # Generate audio
        audio = generate_music(
            prompt=prompt,
            duration=duration,
            temperature=temperature,
            language=language,
            instrumental=instrumental
        )

        # Generate unique filename
        job_id = job.get("id", str(uuid.uuid4())[:8])
        filename = f"heartmula_{job_id}_{uuid.uuid4().hex[:8]}.wav"

        # Upload to S3
        audio_url = upload_to_s3(audio, SAMPLE_RATE, filename)

        return {
            "status": "completed",
            "audio_url": audio_url,
            "duration": duration,
            "prompt": prompt[:100],
            "language": language,
            "sample_rate": SAMPLE_RATE
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


# For local testing
if __name__ == "__main__":
    test_input = {
        "input": {
            "prompt": "A melodic Tamil folk song about nature",
            "duration": 15,
            "language": "tamil"
        }
    }
    result = handler(test_input)
    print(result)


runpod.serverless.start({"handler": handler})
