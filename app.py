import os
import io
import csv
import json
import requests
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify
from google.cloud import bigquery

# import your pipeline as-is
import pipeline

app = Flask(__name__)

# BigQuery config via env vars
BQ_DATASET = os.environ.get("BIGQUERY_DATASET", "my_app_ds")
BQ_TABLE_DETAILS = os.environ.get("BIGQUERY_TABLE_DETAILS", "positive_monthly")
BQ_TABLE_SUM_BY_CAT = os.environ.get("BIGQUERY_TABLE_SUM_BY_CAT", "summary_by_category")
BQ_TABLE_SUM_BY_MAIN = os.environ.get("BIGQUERY_TABLE_SUM_BY_MAIN", "summary_by_main")
BQ_TABLE_TOTAL_POS_BY_MONTH = os.environ.get("BIGQUERY_TABLE_TOTAL_POS_BY_MONTH", "total_positive_by_month")
BQ_TABLE_YEAR_MAIN_TOTALS = os.environ.get("BIGQUERY_TABLE_YEAR_MAIN_TOTALS", "year_main_totals")

# GitHub config via env vars
GITHUB_OWNER = os.environ.get("GITHUB_OWNER")      # e.g., "my-org"
GITHUB_REPO = os.environ.get("GITHUB_REPO")        # e.g., "my-repo"
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main")
GITHUB_DATA_PATH = os.environ.get("GITHUB_DATA_PATH")   # e.g., "data/input.csv"
GITHUB_CONFIG_PATH = os.environ.get("GITHUB_CONFIG_PATH")  # e.g., "config/categories.json"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")  # injected via Secret Manager for private repos

client = bigquery.Client()

def fetch_github_file(path, branch=GITHUB_BRANCH):
    if not GITHUB_OWNER or not GITHUB_REPO or not path:
        raise ValueError("GitHub settings missing: GITHUB_OWNER, GITHUB_REPO, and path are required.")
    # If token is present, use GitHub API to access private repos; else use raw for public
    if GITHUB_TOKEN:
        url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/{path}?ref={branch}"
        headers = {
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.raw+json"
        }
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        return r.text
    else:
        url = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{branch}/{path}"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.text

def classifier_from_github_config():
    if not GITHUB_CONFIG_PATH:
        raise ValueError("GITHUB_CONFIG_PATH not set")
    content = fetch_github_file(GITHUB_CONFIG_PATH)
    # save to a temp file because your loader expects a path
    tmp = "/tmp/categories.json"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    return pipeline.make_classifier_grouped(tmp)

def to_records_date_safe(df):
    # convert Period to string for JSON/BigQuery
    out = df.copy()
    if "Month" in out.columns:
        out["Month"] = out["Month"].astype(str)
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return out.to_dict(orient="records")

def insert_rows(table, rows):
    if not rows:
        return
    table_id = f"{client.project}.{BQ_DATASET}.{table}"
    # Convert any non-JSON-serializable types
    clean_rows = []
    for r in rows:
        cr = {}
        for k, v in r.items():
            if isinstance(v, (pd.Period,)):
                cr[k] = str(v)
            elif isinstance(v, (pd.Timestamp,)):
                cr[k] = v.isoformat()
            else:
                cr[k] = v
        clean_rows.append(cr)
    errors = client.insert_rows_json(table_id, clean_rows)
    if errors:
        raise RuntimeError(f"BigQuery insert errors for {table}: {errors}")

@app.get("/")
def index():
    return "Service is up. Try GET /health or POST /run"

@app.get("/health")
def health():
    return "OK"

@app.post("/run")
def run():
    """
    Body (JSON) options:
    {
      "data_path": "data/input.csv",        # overrides GITHUB_DATA_PATH
      "config_path": "config/categories.json", # overrides GITHUB_CONFIG_PATH
      "card_name": "Uploaded"
    }
    """
    body = request.get_json(silent=True) or {}
    data_path = body.get("data_path") or GITHUB_DATA_PATH
    config_path = body.get("config_path") or GITHUB_CONFIG_PATH
    card_name = body.get("card_name", "Uploaded")
    if not data_path or not config_path:
        return jsonify(error="data_path and config_path are required"), 400

    # Fetch files from GitHub
    data_text = fetch_github_file(data_path)
    cfg_text = fetch_github_file(config_path)
    # write config to temp path for pipeline
    cfg_tmp = "/tmp/categories.json"
    with open(cfg_tmp, "w", encoding="utf-8") as f:
        f.write(cfg_text)

    # Build classifier from config
    classify = pipeline.make_classifier_grouped(cfg_tmp)

    # Parse CSV into DataFrame
    df = pd.read_csv(io.StringIO(data_text))
    df = pipeline.clean_and_standardize(df, card_name=card_name)

    # Execute pipeline
    result = pipeline.run_pipeline([df], classify)

    # Prepare DataFrames for BigQuery
    df_pos_monthly = pd.DataFrame(result["all_positive_monthly"])
    df_sum_by_cat = pd.DataFrame(result["summary_by_category"])
    df_sum_by_main = pd.DataFrame(result["summary_by_main"])
    df_total_pos_by_month = pd.DataFrame(result["total_positive_by_month"])
    df_year_main_totals = pd.DataFrame(result["latest_year_main_totals"])

    # Insert to BigQuery
    insert_rows(BQ_TABLE_DETAILS, to_records_date_safe(df_pos_monthly))
    insert_rows(BQ_TABLE_SUM_BY_CAT, to_records_date_safe(df_sum_by_cat))
    insert_rows(BQ_TABLE_SUM_BY_MAIN, to_records_date_safe(df_sum_by_main))
    insert_rows(BQ_TABLE_TOTAL_POS_BY_MONTH, to_records_date_safe(df_total_pos_by_month))
    insert_rows(BQ_TABLE_YEAR_MAIN_TOTALS, to_records_date_safe(df_year_main_totals))

    # Return summary
    return jsonify({
        "latest_month": str(result["latest_month"]),
        "latest_year": result["latest_year"],
        "inserted": {
            "details": len(df_pos_monthly),
            "summary_by_category": len(df_sum_by_cat),
            "summary_by_main": len(df_sum_by_main),
            "total_positive_by_month": len(df_total_pos_by_month),
            "year_main_totals": len(df_year_main_totals),
        }
    })
