FROM python:3.11-slim

WORKDIR /app

# Install system deps for pandas/pyarrow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
CMD exec gunicorn --bind :$PORT --workers 2 --threads 8 app:app
