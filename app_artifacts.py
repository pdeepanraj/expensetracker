# app_artifacts.py snippet
import os, json, io
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from google.cloud import storage

BUCKET = os.environ["GCS_BUCKET"]
ART_PREFIX = os.environ.get("GCS_ART_PREFIX", "artifacts/")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

def gcs_text(path: str) -> str:
    client = storage.Client()
    bucket = client.bucket(BUCKET)
    blob = bucket.blob(path)
    return blob.download_as_text()

@app.get("/dashboard")
def dashboard(request: Request):
    summary_text = gcs_text(f"{ART_PREFIX.rstrip('/')}/summaries.json")
    summary = json.loads(summary_text)
    return templates.TemplateResponse("dashboard.html", {"request": request, "summary": summary})

@app.get("/rows.jsonl")
def rows_stream():
    client = storage.Client()
    bucket = client.bucket(BUCKET)
    blob = bucket.blob(f"{ART_PREFIX.rstrip('/')}/rows.jsonl")
    buf = io.BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    return StreamingResponse(buf, media_type="application/json")
