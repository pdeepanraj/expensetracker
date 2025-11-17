FROM python:3.11-slim

WORKDIR /app

# Install deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy code and assets
COPY app/ ./app/
COPY static/ ./static/

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# If you serve static files via FastAPI, ensure main.py mounts /static
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
