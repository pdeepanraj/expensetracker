# app.py
import io
import os
import json
import requests
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for

from google.cloud import bigquery
from pipeline import make_classifier_grouped, clean_and_standardize, run_pipeline

app = Flask(__name__, template_folder="templates")

# Environment variables expected:
#   BQ_PROJECT = your GCP project ID (e.g., "deepanexpense")
#   BQ_DATASET = BigQuery dataset (e.g., "expense_analytics")
#   GITHUB_TOKEN (optional) for private repos access
BQ_PROJECT = os.environ.get("BQ_PROJECT")
BQ_DATASET = os.environ.get("BQ_DATASET", "expense_analytics")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")  # optional

# ---------------- GitHub helpers ----------------
def gh_headers():
    h = {"Accept": "application/vnd.github.v3+json", "User-Agent": "expense-tracker-app"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"token {GITHUB_TOKEN}"
    return h

def gh_list_dir(owner: str, repo: str, path: str | None, ref: str | None):
    # GET /repos/{owner}/{repo}/contents/{path}?ref=branch
    base = f"https://api.github.com/repos/{owner}/{repo}/contents"
    url = f"{base}/{path.strip('/')}" if path else base
    params = {"ref": ref} if ref else {}
    r = requests.get(url, headers=gh_headers(), params=params, timeout=30)
    if r.status_code == 404:
        return None, "Path not found."
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and data.get("type") == "file":
        return [data], None
    if isinstance(data, list):
        return data, None
    return [], None

def gh_fetch_raw(owner: str, repo: str, path: str, ref: str):
    # Raw content with branch/ref
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"
    r = requests.get(url, headers=gh_headers(), timeout=60)
    r.raise_for_status()
    return r.content

# ---------------- Ingest helpers ----------------
def fetch_csv_to_df_bytes(b: bytes) -> pd.DataFrame:
    # Try CSV first, fallback to Excel if someone uploaded XLSX
    try:
        return pd.read_csv(io.BytesIO(b))
    except Exception:
        try:
            return pd.read_excel(io.BytesIO(b))
        except Exception as e:
            raise ValueError(f"Unable to parse file as CSV or Excel: {e}")

def load_classifier_from_repo(owner: str, repo: str, ref: str, json_path: str | None):
    # If json_path provided, load categories_grouped.json from repo; else use bundled file
    if json_path:
        b = gh_fetch_raw(owner, repo, json_path.strip("/"), ref)
        cfg = json.loads(b.decode("utf-8"))
        tmp_path = "/tmp/categories_grouped.json"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        return make_classifier_grouped(tmp_path)
    else:
        return make_classifier_grouped("categories_grouped.json")

# ---------------- BigQuery helpers ----------------
def bq_client() -> bigquery.Client:
    proj = BQ_PROJECT
    ds = BQ_DATASET
    print(f"[BOOT] BQ_PROJECT={proj}, BQ_DATASET={ds}")
    if not proj or not proj.strip():
        raise RuntimeError("Missing env var BQ_PROJECT")
    if not ds or not ds.strip():
        raise RuntimeError("Missing env var BQ_DATASET")
    return bigquery.Client(project=proj)

def validate_dataset(project_id: str, dataset_id: str) -> bool:
    client = bq_client()
    ds_ref = bigquery.DatasetReference(project_id, dataset_id)
    try:
        ds = client.get_dataset(ds_ref)
        print(f"[DATASET] Found {project_id}.{dataset_id} location={ds.location}")
        return True
    except Exception as e:
        print(f"[DATASET] Not found or inaccessible {project_id}.{dataset_id}: {e}")
        return False

def bq_query(sql: str, params: dict | None = None) -> list[dict]:
    client = bq_client()
    job_config = bigquery.QueryJobConfig()
    if params:
        job_config.query_parameters = [
            bigquery.ScalarQueryParameter(k, "STRING", str(v)) for k, v in params.items()
        ]
    job = client.query(sql, job_config=job_config)
    rows = list(job.result())
    return [dict(row) for row in rows]

# Only one table we load: all_positive_monthly
SCHEMA_ALL_POSITIVE_MONTHLY = [
    bigquery.SchemaField("Month", "STRING"),
    bigquery.SchemaField("CardName", "STRING"),
    bigquery.SchemaField("MainCategory", "STRING"),
    bigquery.SchemaField("Category", "STRING"),
    bigquery.SchemaField("Description", "STRING"),
    bigquery.SchemaField("Amount", "FLOAT"),
    bigquery.SchemaField("Comment", "STRING"),
]

def ensure_table(table_name: str, schema: list[bigquery.SchemaField]):
    client = bq_client()
    dataset_ref = bigquery.DatasetReference(BQ_PROJECT, BQ_DATASET)
    table_ref = dataset_ref.table(table_name)
    try:
        client.get_table(table_ref)
        print(f"[TABLE] Exists: {table_ref.table_id}")
    except Exception:
        table = bigquery.Table(table_ref, schema=schema)
        client.create_table(table)
        print(f"[TABLE] Created: {table_ref.table_id}")

def load_records(table_name: str, records: list[dict], schema: list[bigquery.SchemaField]):
    if not records:
        print(f"[LOAD] Skip empty load to {table_name}")
        return
    client = bq_client()
    table_id = f"{BQ_PROJECT}.{BQ_DATASET}.{table_name}"
    job_config = bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_TRUNCATE")
    print(f"[LOAD] Loading {len(records)} rows into {table_id}")
    job = client.load_table_from_json(records, table_id, job_config=job_config)
    job.result()
    print(f"[LOAD] Completed {table_id}")

# ---------------- Routes ----------------
@app.get("/")
def index():
    return render_template("index.html")

@app.get("/list")
def list_csvs():
    owner = request.args.get("owner", "").strip()
    repo = request.args.get("repo", "").strip()
    branch = request.args.get("branch", "").strip() or "main"
    path = request.args.get("path", "").strip()
    if not owner or not repo:
        return redirect(url_for("index"))

    items, err = gh_list_dir(owner, repo, path, branch)
    files = []
    if not err and items:
        for it in items:
            if it.get("type") == "file" and it.get("name", "").lower().endswith(".csv"):
                files.append({"name": it["name"], "path": it["path"]})
    return render_template("list.html", owner=owner, repo=repo, branch=branch, path=path, files=files, error=err)

@app.post("/process")
def process():
    owner = request.form.get("owner", "").strip()
    repo = request.form.get("repo", "").strip()
    branch = request.form.get("branch", "").strip() or "main"
    path = request.form.get("path", "").strip()
    csv_paths = request.form.getlist("csv_paths")  # repo-relative paths
    json_path = request.form.get("json_path", "").strip() or None
    card_name = request.form.get("card_name", "Uploaded").strip() or "Uploaded"

    if not owner or not repo or not csv_paths:
        return "Please select at least one CSV.", 400

    try:
        if not validate_dataset(BQ_PROJECT, BQ_DATASET):
            return f"Error: Dataset {BQ_PROJECT}.{BQ_DATASET} not found or not accessible.", 500

        classifier = load_classifier_from_repo(owner, repo, branch, json_path)

        frames = []
        for p in csv_paths:
            b = gh_fetch_raw(owner, repo, p, branch)
            df = fetch_csv_to_df_bytes(b)
            df_clean = clean_and_standardize(df, card_name=card_name)
            frames.append(df_clean)

        # Run your patched pipeline (outputs JSON-safe dicts)
        result = run_pipeline(frames, classifier)

        # Create and load only all_positive_monthly
        ensure_table("all_positive_monthly", SCHEMA_ALL_POSITIVE_MONTHLY)
        records = result["all_positive_monthly"]
        load_records("all_positive_monthly", records, SCHEMA_ALL_POSITIVE_MONTHLY)

        # Redirect to dashboard for the latest month just processed
        latest_month = str(result.get("latest_month"))
        return redirect(url_for("dashboard", month=latest_month), code=303)

    except Exception as e:
        return f"Error: {type(e).__name__}: {e}", 500

@app.get("/dashboard")
def dashboard():
    if not validate_dataset(BQ_PROJECT, BQ_DATASET):
        return f"Error: Dataset {BQ_PROJECT}.{BQ_DATASET} not accessible.", 500

    table_id = f"`{BQ_PROJECT}.{BQ_DATASET}.all_positive_monthly`"
    selected_month = request.args.get("month", "").strip() or None

    # Determine latest month available
    latest_month_sql = f"""
      SELECT Month
      FROM {table_id}
      WHERE Month IS NOT NULL
      ORDER BY Month DESC
      LIMIT 1
    """
    latest_month_rows = bq_query(latest_month_sql)
    if not latest_month_rows:
        return "No data available yet.", 200
    latest_month = latest_month_rows[0]["Month"]

    month_for_view = selected_month or latest_month

    # Top categories for the selected month
    top_categories_sql = f"""
      SELECT Category, SUM(Amount) AS Amount
      FROM {table_id}
      WHERE Month = @month
      GROUP BY Category
      ORDER BY Amount DESC
      LIMIT 12
    """
    top_categories = bq_query(top_categories_sql, params={"month": month_for_view})

    # Monthly totals trend
    monthly_totals_sql = f"""
      SELECT Month, SUM(Amount) AS Amount
      FROM {table_id}
      GROUP BY Month
      ORDER BY Month
    """
    monthly_totals = bq_query(monthly_totals_sql)

    # Details for the selected month
    latest_detail_sql = f"""
      SELECT CardName, MainCategory, Category, Description, Amount
      FROM {table_id}
      WHERE Month = @month
      ORDER BY Amount DESC
      LIMIT 1000
    """
    latest_details = bq_query(latest_detail_sql, params={"month": month_for_view})

    return render_template(
        "dashboard.html",
        latest_month=latest_month,           # absolute latest
        month_for_view=month_for_view,       # chosen or latest
        top_categories=top_categories,
        monthly_totals=monthly_totals,
        latest_details=latest_details,
        project=BQ_PROJECT,
        dataset=BQ_DATASET,
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
