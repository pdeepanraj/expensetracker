FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app package, plus static/templates
COPY app/ ./app/
COPY static/ ./static/
COPY app/templates/ ./app/templates/

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
