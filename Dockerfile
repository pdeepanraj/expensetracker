# Use Python base
FROM python:3.11-slim

# Create app dir
WORKDIR /app

# Install system deps (optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates && rm -rf /var/lib/apt/lists/*

# Copy and install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Set port for Cloud Run
ENV PORT=8080
# Flask
CMD exec gunicorn --bind :$PORT --workers 2 --threads 8 app:app
