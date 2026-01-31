"""
HeartMuLa 3B RunPod Serverless Handler
Tamil/Multilingual Music Generation with S3 Output Storage

Uses the official heartlib library for generation.

Compatible with the HeartMuLaProvider API format:
- lyrics: Structured lyrics with [Verse], [Chorus] etc.
- tags: Comma-separated style tags (no spaces between tags)
- max_duration_seconds: Duration in seconds
- temperature, topk, cfg_scale: Generation parameters
"""

import runpod
import torch
import os
import uuid
import boto3
import tempfile

# S3 Configuration
S3_BUCKET = os.environ.get("S3_BUCKET", "aiswara-music")
S3_REGION = os.environ.get("S3_REGION", "us-east-1")
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

# Model configuration
MODEL_PATH = os.environ.get("HEARTMULA_MODEL_PATH", "/app/ckpt")
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


def upload_to_s3(audio_path: str, filename: str) -> str:
    """Upload audio file to S3 and return streaming URL"""
    s3 = get_s3_client()

    key = f"generated/{filename}"

    # Determine content type
    content_type = 'audio/mpeg' if filename.endswith('.mp3') else 'audio/wav'

    s3.upload_file(
        audio_path,
        S3_BUCKET,
        key,
        ExtraArgs={
            'ContentType': content_type,
            'ACL': 'public-read'
        }
    )

    # Return public URL
    return f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{key}"


def load_model():
    """Load HeartMuLa 3B model using heartlib"""
    global pipeline

    if pipeline is not None:
        return pipeline

    print("Loading HeartMuLa 3B model with heartlib...")

    try:
        from heartlib import HeartMuLaGenPipeline

        # Determine device and dtype
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device = {
            "mula": torch.device(device_str),
            "codec": torch.device(device_str),
        }
        dtype = torch.float16 if device_str == "cuda" else torch.float32

        pipeline = HeartMuLaGenPipeline.from_pretrained(
            MODEL_PATH,
            device=device,
            dtype=dtype,
            version="3B",
            lazy_load=False
        )
        print("HeartMuLa model loaded successfully!")

    except ImportError as e:
        print(f"heartlib import error: {e}")
        raise RuntimeError(f"heartlib not installed correctly: {e}")
    except Exception as e:
        print(f"Model loading error: {e}")
        raise

    return pipeline


def generate_music(lyrics: str, tags: str, duration: int = 120,
                   temperature: float = 1.0, topk: int = 50,
                   cfg_scale: float = 1.5) -> str:
    """Generate music from lyrics and tags, returns path to output file"""

    pipe = load_model()

    # Create temp directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write lyrics and tags to files (heartlib reads from files)
        lyrics_file = os.path.join(tmpdir, "lyrics.txt")
        tags_file = os.path.join(tmpdir, "tags.txt")
        output_file = os.path.join(tmpdir, "output.mp3")

        with open(lyrics_file, 'w') as f:
            f.write(lyrics)

        # Tags should be comma-separated without spaces
        clean_tags = ','.join([t.strip() for t in tags.split(',')])
        with open(tags_file, 'w') as f:
            f.write(clean_tags)

        # Convert duration to milliseconds
        max_audio_length_ms = duration * 1000

        # Generate music using the pipeline
        # The pipeline expects a dict with paths
        pipe(
            {
                "lyrics_path": lyrics_file,
                "tags_path": tags_file,
            },
            output_path=output_file,
            max_audio_length_ms=max_audio_length_ms,
            topk=topk,
            temperature=temperature,
            cfg_scale=cfg_scale
        )

        # Read the generated file and return a permanent copy
        output_bytes = open(output_file, 'rb').read()

    # Save to a persistent temp file
    final_output = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
    final_output.write(output_bytes)
    final_output.close()

    return final_output.name


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
        audio_path = generate_music(
            lyrics=lyrics,
            tags=tags,
            duration=duration,
            temperature=temperature,
            topk=topk,
            cfg_scale=cfg_scale
        )

        # Generate unique filename
        job_id = job.get("id", str(uuid.uuid4())[:8])
        filename = f"heartmula_{job_id}_{uuid.uuid4().hex[:8]}.mp3"

        # Upload to S3
        audio_url = upload_to_s3(audio_path, filename)

        # Clean up temp file
        os.unlink(audio_path)

        return {
            "status": "completed",
            "audio_url": audio_url,
            "duration": duration,
            "tags": tags,
            "sample_rate": SAMPLE_RATE,
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
