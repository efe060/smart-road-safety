FROM nvidia/cuda:12.1.0-base-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN mkdir -p /app/data/input /app/data/output

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY main.py .

# Mount a video at /app/data/input/video.mp4; the result is written to
# /app/data/output/results.json
CMD ["python3", "main.py"]
