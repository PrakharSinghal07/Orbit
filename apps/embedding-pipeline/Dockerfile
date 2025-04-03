FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY embedding-pipeline/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY embedding-pipeline/ /app/embedding-pipeline/

ENV PYTHONPATH=/app

CMD ["python", "embedding-pipeline/cmd/main/main.py"]