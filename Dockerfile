FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline.py categories_grouped.json app.py ./
COPY templates/ ./templates/

ENV PORT=8080
CMD ["python", "app.py"]
