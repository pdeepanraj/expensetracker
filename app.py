# app.py
import io
import os
import json
import requests
import pandas as pd
from flask import Flask, request, render_template_string, redirect, url_for
from google.cloud import bigquery

from pipeline import make_classifier_grouped, clean_and_standardize, run_pipeline

app = Flask(__name__)

BQ_PROJECT = os.environ.get("BQ_PROJECT")              # e.g., "your-gcp-project-id"
BQ_DATASET = os.environ.get("BQ_DATASET", "expense_tracker")

HTML_FORM = """
<!doctype html>
<title>Expense Tracker Processor</title>
<h2>Process Expenses from GitHub</h2>
<form method="post" action="/process">
  <label>Card Name (for tagging):</label>
  <input type="text" name="card_name" placeholder="Uploaded" value="Uploaded" required><br><br>

  <label>Category JSON raw URL (default uses local categories_grouped.json):</label>
  <input type="url" name="json_url" placeholder="https://raw.githubusercontent.com/.../categories_grouped.json"><br><br>

  <label>CSV raw URLs (one per line):</label><br>
  <textarea name="csv_urls" rows="8" cols="80" placeholder="https://raw.githubusercontent.com/.../file1.csv
https://raw.githubusercontent.com/.../file2.csv"></textarea><br><br>

  <button type="submit">Run Pipeline</button>
</form>
"""

def fetch_text(url: str) -> str:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.text

def fetch_csv_to_df(url: str) -> pd.DataFrame:
    txt = fetch_text(url)
    # Try CSV first, fallback to Excel if needed (optional)
    try:
        return pd.read_csv(io.StringIO(txt))
    except Exception:
        # If someone uploaded an XLSX via Git LFS, you’d fetch bytes and use read_excel
        raise

def load_classifier(json_url: str | None):
    # If no json_url, use local categories_grouped.json bundled with the container
    if json_url:
        cfg_str = fetch_text(json_url)
        cfg = json.loads(cfg_str)
        # Save to a temp file for make_classifier_grouped which expects a path
        tmp_path = "/tmp/categories_grouped.json"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        return make_classifier_grouped(tmp_path)
    else:
        return make_classifier_grouped("categories_grouped.json")

def bq_client() -> bigquery.Client:
    return bigquery.Client(project=BQ_PROJECT)

def ensure_tables(schema_map: dict):
    client = bq_client()
    dataset_ref = bigquery.DatasetReference(BQ_PROJECT, BQ_DATASET)
    for table_name, schema in schema_map.items():
        table_ref = dataset_ref.table(table_name)
        try:
            client.get_table(table_ref)
        except Exception:
            table = bigquery.Table(table_ref, schema=schema)
            client.create_table(table)

def load_records(table_name: str, records: list[dict], schema: list[bigquery.SchemaField]):
    if not records:
        return
    client = bq_client()
    table_id = f"{BQ_PROJECT}.{BQ_DATASET}.{table_name}"
    job = client.load_table_from_json(records, table_id, job_config=bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_TRUNCATE"))
    job.result()

@app.get("/")
def index():
    return render_template_string(HTML_FORM)

@app.post("/process")
def process():
    card_name = request.form.get("card_name", "Uploaded")
    json_url = request.form.get("json_url", "").strip() or None
    csv_urls_raw = request.form.get("csv_urls", "")
    csv_urls = [u.strip() for u in csv_urls_raw.splitlines() if u.strip()]
    if not csv_urls:
        return "Please provide at least one CSV raw URL from GitHub.", 400

    try:
        classifier = load_classifier(json_url)
        frames = []
        for url in csv_urls:
            df = fetch_csv_to_df(url)
            df_clean = clean_and_standardize(df, card_name=card_name)
            frames.append(df_clean)

        result = run_pipeline(frames, classifier)

        # Define schemas
        schema_positive_monthly = [
            bigquery.SchemaField("Month", "STRING"),
            bigquery.SchemaField("CardName", "STRING"),
            bigquery.SchemaField("MainCategory", "STRING"),
            bigquery.SchemaField("Category", "STRING"),
            bigquery.SchemaField("Description", "STRING"),
            bigquery.SchemaField("Amount", "FLOAT"),
            bigquery.SchemaField("Comment", "STRING"),
        ]
        schema_all_positive_monthly = list(schema_positive_monthly)
        schema_summary_by_category = [
            bigquery.SchemaField("Month", "STRING"),
            bigquery.SchemaField("Category", "STRING"),
            bigquery.SchemaField("Amount", "FLOAT"),
        ]
        schema_summary_by_main = [
            bigquery.SchemaField("Month", "STRING"),
            bigquery.SchemaField("MainCategory", "STRING"),
            bigquery.SchemaField("Amount", "FLOAT"),
        ]
        schema_total_positive_by_month = [
            bigquery.SchemaField("Month", "STRING"),
            bigquery.SchemaField("Amount", "FLOAT"),
        ]
        schema_latest_year_main_totals = [
            bigquery.SchemaField("Year", "INTEGER"),
            bigquery.SchemaField("MainCategory", "STRING"),
            bigquery.SchemaField("Amount", "FLOAT"),
            bigquery.SchemaField("Subcategories", "STRING"),
        ]

        schema_map = {
            "positive_monthly": schema_positive_monthly,
            "all_positive_monthly": schema_all_positive_monthly,
            "summary_by_category": schema_summary_by_category,
            "summary_by_main": schema_summary_by_main,
            "total_positive_by_month": schema_total_positive_by_month,
            "latest_year_main_totals": schema_latest_year_main_totals,
        }
        ensure_tables(schema_map)

        # Load
        load_records("positive_monthly", result["positive_monthly"], schema_positive_monthly)
        load_records("all_positive_monthly", result["all_positive_monthly"], schema_all_positive_monthly)
        load_records("summary_by_category", result["summary_by_category"], schema_summary_by_category)
        load_records("summary_by_main", result["summary_by_main"], schema_summary_by_main)
        load_records("total_positive_by_month", result["total_positive_by_month"], schema_total_positive_by_month)
        load_records("latest_year_main_totals", result["latest_year_main_totals"], schema_latest_year_main_totals)

        # Simple confirmation page
        latest_month = str(result.get("latest_month"))
        latest_year = result.get("latest_year")
        return f"Pipeline completed. Latest month: {latest_month}, Latest year: {latest_year}. Data loaded to BigQuery dataset {BQ_DATASET}."
    except Exception as e:
        return f"Error: {e}", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
