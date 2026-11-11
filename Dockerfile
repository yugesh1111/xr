# Inference image for the submission: predict() / predictions.jsonl on CPU, no network at run time.
#   docker build -t echogesture .
#   docker run --rm --network none -v /path/to/clips:/clips -v "$PWD/results:/app/results" \
#       echogesture --videos /clips --out results/predictions.jsonl
FROM python:3.11-slim

# MediaPipe needs EGL/GLES even on CPU; OpenCV needs libgl
RUN apt-get update && apt-get install -y --no-install-recommends libegl1 libgles2 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

COPY echogesture/ echogesture/
COPY models/ models/
COPY main.py .
# bake the MediaPipe Holistic model into the image so runs work with --network none
RUN python -c "from echogesture.extract import MODEL_URL, DEFAULT_MODEL; import urllib.request; \
DEFAULT_MODEL.exists() or urllib.request.urlretrieve(MODEL_URL, DEFAULT_MODEL)"

ENTRYPOINT ["python", "main.py"]
