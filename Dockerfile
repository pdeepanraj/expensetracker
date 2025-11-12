FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY gcs_ingest.py .
CMD ["python", "gcs_ingest.py"]
