FROM python:3.10-slim

WORKDIR /app

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Preload the model using an alternative approach
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('intfloat/multilingual-e5-large-instruct', local_dir='/app/model')"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('/app/model')"

# Copy your app
COPY . .

# Set environment variables
ENV PYTHONPATH=/app

# Expose and run
EXPOSE 8000
CMD ["python", "api/main.py"]