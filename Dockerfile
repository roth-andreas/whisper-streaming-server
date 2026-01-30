# Whisper Streaming Server - Docker Image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies (git for pip packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Expose WebSocket port
EXPOSE 9000

# Environment variables
ENV WHISPER_MODEL_PATH=turbo.pt
ENV WHISPER_LAN=en
ENV PYTHONUNBUFFERED=1

# Start server
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "9000"]
