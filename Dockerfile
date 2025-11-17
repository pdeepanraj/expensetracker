FROM python:3.11-slim

WORKDIR /app

COPY app/ ./app/
COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
