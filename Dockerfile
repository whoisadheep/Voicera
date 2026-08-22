FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1



# Install ffmpeg (needed for Sarvam MP3 → PCM decoding)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway/Render set PORT env var automatically
CMD uvicorn exotel_server:app --host 0.0.0.0 --port $PORT
