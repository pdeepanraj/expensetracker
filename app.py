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

BQ_PROJECT = os.environ.get("BQ_PROJECT")
BQ_DATASET = os.environ.get("BQ_DATASET", "expense_tracker")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")  # optional, for private repos

def gh_headers():
    h = {"Accept": "application/vnd.github.v3+json", "User-Agent": "expense-tracker-app"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"token {GITHUB_TOKEN}"
    return h

def gh_list_dir(owner: str, repo: str, path: str | None, ref: str | None):
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
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"
    r = requests.get(url, headers=gh_headers(), timeout=60)
    r.raise_for_status()
    return r.content

def fetch_csv_to_df_bytes(b: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(b))
    except Exception:
        try:
            return pd.read_excel(io.BytesIO(b))
        except Exception as e:
            raise ValueError(f"Unable to parse file as CSV or Excel: {e}")

def load_classifier_from_repo(owner: str, repo: str, ref: str, json_path: str | None):
    if json_path:
        b = gh_fetch_raw(owner, repo, json_path.strip("/"), ref)
        cfg = json.loads(b.decode("utf-8"))
        tmp_path = "/tmp/categories_grouped.json"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        return make_classifier_grouped(tmp_path)
    else:
        return make_classifier_grouped("categories_grouped.json")

def bq_client() -> bigquery.Client:
    proj = os.environ.get("BQ_PROJECT")
    ds = os.environ.get("BQ_DATASET")
    print(f"[BOOT] BQ_PROJECT={proj}, BQ_DATASET={ds}")
    if not proj or not proj.strip():
        raise RuntimeError("Missing env var BQ_PROJECT")
    if not ds or not ds.strip():
        raise RuntimeError("Missing env var BQ_DATASET")
    return bigquery.Client(project=proj)

def validate_dataset(project_id: str, dataset_id: str):
    client = bq_client()
    ds_ref = bigquery.DatasetReference(project_id, dataset_id)
    try:
        ds = client.get_dataset(ds_ref)
        print(f"[DATASET] Found {project_id}.{dataset_id} location={ds.location}")
        return True
    except Exception as e:
        print(f"[DATASET] Not found {project_id}.{dataset_id}: {e}")
        return False

def ensure_tables(schema_map: dict):
    client = bq_client()
    dataset_ref = bigquery.DatasetReference(os.environ["BQ_PROJECT"], os.environ["BQ_DATASET"])
    for table_name, schema in schema_map.items():
        table_ref = dataset_ref.table(table_name)
        try:
            client.get_table(table_ref)
            print(f"[TABLE] Exists: {table_ref.table_id}")
        except Exception:
            try:
                table = bigquery.Table(table_ref, schema=schema)
                client.create_table(table)
                print(f"[TABLE] Created: {table_ref.table_id}")
            except Exception as e:
                print(f"[TABLE] Create failed {table_ref.table_id}: {type(e).__name__}: {e}")
                raise

def load_records(table_name: str, records: list[dict], schema: list[bigquery.SchemaField]):
    if not records:
        print(f"[LOAD] Skip empty load to {table_name}")
        return
    client = bq_client()
    table_id = f"{os.environ['BQ_PROJECT']}.{os.environ['BQ_DATASET']}.{table_name}"
    job_config = bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_TRUNCATE")
    print(f"[LOAD] Loading {len(records)} rows into {table_id}")
    try:
        job = client.load_table_from_json(records, table_id, job_config=job_config)
        job.result()
        print(f"[LOAD] Completed {table_id}")
    except Exception as e:
        # Surface full error chain
        try:
            errs = job.errors if 'job' in locals() else None
        except Exception:
            errs = None
        print(f"[LOAD] Failed {table_id}: {type(e).__name__}: {e}; details={errs}")
        raise
        
def ensure_dataset(project_id: str, dataset_id: str, location: str = "US"):
    client = bigquery.Client(project=project_id)
    dataset_ref = bigquery.DatasetReference(project_id, dataset_id)
    try:
        client.get_dataset(dataset_ref)
    except Exception:
        ds = bigquery.Dataset(dataset_ref)
        ds.location = location
        client.create_dataset(ds)



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
    csv_paths = request.form.getlist("csv_paths")
    json_path = request.form.get("json_path", "").strip() or None
    card_name = request.form.get("card_name", "Uploaded").strip() or "Uploaded"

    if not owner or not repo or not csv_paths:
        return "Please select at least one CSV.", 400

    try:
        classifier = load_classifier_from_repo(owner, repo, branch, json_path)
        frames = []
        for p in csv_paths:
            b = gh_fetch_raw(owner, repo, p, branch)
            df = fetch_csv_to_df_bytes(b)
            df_clean = clean_and_standardize(df, card_name=card_name)
            frames.append(df_clean)

        result = run_pipeline(frames, classifier)

        from google.cloud import bigquery
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
        ensure_dataset(BQ_PROJECT, BQ_DATASET, location="US")
        # before ensure_tables(schema_map)
        if not validate_dataset(BQ_PROJECT, BQ_DATASET):
            return f"Error: Dataset {BQ_PROJECT}.{BQ_DATASET} not found or not accessible.", 500
        ensure_tables(schema_map)

        load_records("positive_monthly", result["positive_monthly"], schema_positive_monthly)
        load_records("all_positive_monthly", result["all_positive_monthly"], schema_all_positive_monthly)
        load_records("summary_by_category", result["summary_by_category"], schema_summary_by_category)
        load_records("summary_by_main", result["summary_by_main"], schema_summary_by_main)
        load_records("total_positive_by_month", result["total_positive_by_month"], schema_total_positive_by_month)
        load_records("latest_year_main_totals", result["latest_year_main_totals"], schema_latest_year_main_totals)

        latest_month = str(result.get("latest_month"))
        latest_year = result.get("latest_year")
        return f"Pipeline completed for {owner}/{repo}@{branch}. Latest month: {latest_month}, Latest year: {latest_year}. Data loaded to {BQ_PROJECT}.{BQ_DATASET}."
    except Exception as e:
        return f"Error: {e}", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

