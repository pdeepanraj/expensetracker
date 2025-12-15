# app.py
import io
import os
import json, re
import hashlib
import requests
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for
from markupsafe import escape
from typing import List, Dict, Any, Tuple
from pathlib import Path

from google.cloud import bigquery
from pipeline import make_classifier_grouped, clean_and_standardize, run_pipeline

app = Flask(__name__, template_folder="templates")

# ---------------- Jinja filters ----------------
@app.template_filter('currency')
def currency_filter(value):
    try:
        n = float(value or 0)
        return f"${n:,.2f}"
    except Exception:
        return value

@app.template_filter('intgroup')
def int_group_filter(value):
    try:
        n = int(value or 0)
        return f"{n:,}"
    except Exception:
        try:
            n = float(value or 0)
            return f"{int(n):,}"
        except Exception:
            return value

# ---------------- Env ----------------
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

# ---------------- Ingest helpers ----------------
def fetch_csv_to_df_bytes(b: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(b))
    except Exception:
        try:
            return pd.read_excel(io.BytesIO(b))
        except Exception as e:
            raise ValueError(f"Unable to parse file as CSV or Excel: {e}")

def derive_card_name_from_path(path: str) -> str:
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    if "_" in name:
        name = name.rsplit("_", 1)[0]
    return name.replace(" ", "_").upper()

def load_classifier_from_repo(owner: str, repo: str, ref: str, json_path: str | None):
    """
    Signature preserved; if json_path is provided we still fetch it, but we DO NOT
    rely on local JSON. We just pass a dummy path to make_classifier_grouped, which
    now reads from BigQuery regardless of the path value.
    """
    if json_path:
        _ = gh_fetch_raw(owner, repo, json_path.strip("/"), ref)
        return make_classifier_grouped("/dev/null")
    else:
        return make_classifier_grouped("/dev/null")

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

# Target table
TARGET_TABLE = "all_positive_monthly"
TARGET_SCHEMA = [
    bigquery.SchemaField("Month", "STRING"),
    bigquery.SchemaField("CardName", "STRING"),
    bigquery.SchemaField("MainCategory", "STRING"),
    bigquery.SchemaField("Category", "STRING"),
    bigquery.SchemaField("Description", "STRING"),
    bigquery.SchemaField("Amount", "FLOAT"),
    bigquery.SchemaField("Comment", "STRING"),
    bigquery.SchemaField("RowHash", "STRING"),
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

def compute_row_hash(rec: dict) -> str:
    key_fields = ["Month", "CardName", "MainCategory", "Category", "Description", "Amount", "Comment"]
    payload = "|".join(str(rec.get(k, "")).strip() for k in key_fields)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def get_table_metadata(table_name: str):
    client = bq_client()
    table_id = f"{BQ_PROJECT}.{BQ_DATASET}.{table_name}"
    table = client.get_table(table_id)
    return {
        "id": table_id,
        "num_rows": table.num_rows,
        "created": getattr(table, "created", None),
        "modified": getattr(table, "modified", None),
    }

def append_non_duplicates(records: list[dict], table_name: str):
    if not records:
        print("[LOAD] No records to process.")
        return 0
    for r in records:
        r["RowHash"] = compute_row_hash(r)
    client = bq_client()
    dataset = f"{BQ_PROJECT}.{BQ_DATASET}"
    target_id = f"{dataset}.{table_name}"
    staging_id = f"{dataset}.{table_name}_staging"
    ensure_table(table_name, TARGET_SCHEMA)
    staging_schema = TARGET_SCHEMA
    try:
        client.delete_table(staging_id, not_found_ok=True)
    except Exception:
        pass
    client.create_table(bigquery.Table(staging_id, schema=staging_schema))
    print(f"[STAGING] Loading {len(records)} rows into {staging_id}")
    job = client.load_table_from_json(
        records,
        staging_id,
        job_config=bigquery.LoadJobConfig(schema=staging_schema, write_disposition="WRITE_TRUNCATE"),
    )
    job.result()
    dedup_sql = f"""
    INSERT INTO `{target_id}` (Month, CardName, MainCategory, Category, Description, Amount, Comment, RowHash)
    SELECT Month, CardName, MainCategory, Category, Description, Amount, Comment, RowHash
    FROM `{staging_id}` s
    WHERE NOT EXISTS (SELECT 1 FROM `{target_id}` t WHERE t.RowHash = s.RowHash)
    """
    print(f"[DEDUP] Inserting non-duplicate rows into {target_id}")
    qjob = client.query(dedup_sql); qjob.result()
    count_sql = f"""
      SELECT COUNT(*) AS cnt
      FROM `{staging_id}` s
      WHERE NOT EXISTS (SELECT 1 FROM `{target_id}` t WHERE t.RowHash = s.RowHash)
    """
    cnt = bq_query(count_sql)[0]["cnt"]
    print(f"[DEDUP] New rows inserted: {cnt}")
    try:
        client.delete_table(staging_id, not_found_ok=True)
    except Exception:
        pass
    return cnt

# ---------------- Filters helpers ----------------
def get_distinct_filters():
    table_id = f"`{BQ_PROJECT}.{BQ_DATASET}.{TARGET_TABLE}`"
    sql = f"""
    SELECT ARRAY_AGG(DISTINCT Month) AS months,
           ARRAY_AGG(DISTINCT CardName) AS cards,
           ARRAY_AGG(DISTINCT MainCategory) AS mains,
           ARRAY_AGG(DISTINCT Category) AS cats
    FROM {table_id}
    """
    rows = bq_query(sql)
    if not rows:
        return [], [], [], []
    r = rows[0]
    months = sorted([m for m in r.get("months", []) if m], reverse=True)
    cards = sorted([c for c in r.get("cards", []) if c])
    mains = sorted([m for m in r.get("mains", []) if m])
    cats = sorted([c for c in r.get("cats", []) if c])
    return months, cards, mains, cats

def apply_filters_where(params: dict) -> tuple[str, dict]:
    where = []
    qp = {}
    if params.get("month"):
        where.append("Month = @month"); qp["month"] = params["month"]
    if params.get("card"):
        where.append("CardName = @card"); qp["card"] = params["card"]
    if params.get("main"):
        where.append("MainCategory = @main"); qp["main"] = params["main"]
    if params.get("cat"):
        where.append("Category = @cat"); qp["cat"] = params["cat"]
    return ("WHERE " + " AND ".join(where)) if where else "", qp

# ---- Category test & add page (now BigQuery-backed) ----
CATEGORY_TABLE = f"{BQ_PROJECT}.{BQ_DATASET}.category_config"
MONTHLY_SRC = f"{BQ_PROJECT}.{BQ_DATASET}.all_positive_monthly"

def fetch_categories_bq() -> list[tuple[str, str, str, bool | None]]:
    client = bq_client()
    sql = f"""
    SELECT Main, Category, Keyword, UseBoundaries
    FROM `{CATEGORY_TABLE}`
    WHERE Main IS NOT NULL AND Category IS NOT NULL AND Keyword IS NOT NULL
    """
    out = []
    for r in client.query(sql).result():
        ub = None
        try:
            ub = bool(r.UseBoundaries) if r.UseBoundaries is not None else None
        except Exception:
            ub = None
        out.append((r.Main, r.Category, r.Keyword, ub))
    return out

def build_regex_index_bq(use_word_boundaries: bool = True):
    idx = []
    for main, cat, kw, ub in fetch_categories_bq():
        base = (kw or "").strip()
        if not base:
            continue
        if ub is None:
            ub = _default_use_boundaries_for_keyword(base)
        if use_word_boundaries and ub:
            pat = re.compile(rf"\b{re.escape(base)}\b", re.IGNORECASE)
        else:
            pat = re.compile(re.escape(base), re.IGNORECASE)
        idx.append((pat, cat, main))
    return idx

def classify_with_bq_index(text: str, regex_index) -> tuple[str, str]:
    t = (text or "").strip()
    for pattern, cat, main in regex_index:
        if pattern.search(t):
            return cat, main
    return "Other", "Other"

def _default_use_boundaries_for_keyword(kw: str) -> bool:
    return bool(re.fullmatch(r'[\w\s]+', kw.lower()))

def insert_keywords_bq(main: str, category: str, keywords: list[str]):
    client = bq_client()
    sql = f"SELECT Main, Category, Keyword FROM `{CATEGORY_TABLE}`"
    existing_rows = list(client.query(sql).result())
    existing = {(str(r.Main).lower(), str(r.Category).lower(), str(r.Keyword).lower()) for r in existing_rows}
    to_insert = []
    for k in keywords:
        k_norm = k.strip().lower()
        key = (main.lower(), category.lower(), k_norm)
        if k_norm and key not in existing:
            use_boundaries = _default_use_boundaries_for_keyword(k_norm)
            to_insert.append({"Main": main, "Category": category, "Keyword": k_norm, "UseBoundaries": use_boundaries})
    if not to_insert:
        return
    errors = client.insert_rows_json(CATEGORY_TABLE, to_insert)
    if errors:
        raise RuntimeError(f"Insert errors: {errors}")

def fetch_other_monthly(limit: int = 200):
    client = bq_client()
    sql = f"""
    SELECT Month, CardName, Description, Amount
    FROM `{MONTHLY_SRC}`
    WHERE LOWER(IFNULL(MainCategory,'other')) = 'other'
    ORDER BY Month DESC, Amount DESC
    LIMIT {limit}
    """
    return list(client.query(sql).result())

def fetch_categories_by_main() -> dict[str, list[str]]:
    client = bq_client()
    sql = f"""
    SELECT Main, Category
    FROM `{CATEGORY_TABLE}`
    WHERE Main IS NOT NULL AND Category IS NOT NULL
    GROUP BY Main, Category
    """
    by_main: dict[str, set[str]] = {}
    for r in client.query(sql).result():
        m = str(r.Main).strip(); c = str(r.Category).strip()
        if m and c:
            by_main.setdefault(m, set()).add(c)
    return {m: sorted(list(cats)) for m, cats in by_main.items()}

# ---------------- Routes: index/list/process ----------------
@app.get("/")
def index():
    project = BQ_PROJECT or ""
    dataset = BQ_DATASET or ""
    table = TARGET_TABLE or ""
    try:
        meta = get_table_metadata(TARGET_TABLE)
    except Exception:
        meta = {"num_rows": 0, "created": None, "modified": None}
    return render_template("index.html", project=project, dataset=dataset, table=table, meta=meta)

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
            derived_card = derive_card_name_from_path(p)
            df_clean = clean_and_standardize(df, card_name=derived_card)
            frames.append(df_clean)
        result = run_pipeline(frames, classifier)
        records = result["all_positive_monthly"]
        new_rows = append_non_duplicates(records, TARGET_TABLE)
        latest_month = str(result.get("latest_month"))
        return redirect(url_for("dashboard", month=latest_month, loaded=new_rows), code=303)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}", 500

# --- Manual spend / income tables ---
MANUAL_SPEND_TABLE = "manual_spend"
MANUAL_SPEND_SCHEMA = [
    bigquery.SchemaField("Month", "STRING"),
    bigquery.SchemaField("Category", "STRING"),
    bigquery.SchemaField("Description", "STRING"),
    bigquery.SchemaField("Amount", "FLOAT"),
    bigquery.SchemaField("Note", "STRING"),
    bigquery.SchemaField("RowId", "STRING"),
]
MONTHLY_INCOME_TABLE = "monthly_income"
MONTHLY_INCOME_SCHEMA = [
    bigquery.SchemaField("Month", "STRING"),
    bigquery.SchemaField("Source", "STRING"),
    bigquery.SchemaField("Amount", "FLOAT"),
    bigquery.SchemaField("Note", "STRING"),
    bigquery.SchemaField("RowId", "STRING"),
]

def ensure_table_with_schema(table_name: str, schema: list[bigquery.SchemaField]):
    ensure_table(table_name, schema)

def upsert_simple(table_name: str, schema: list[bigquery.SchemaField], rows: list[dict]):
    for r in rows:
        payload = "|".join(str(r.get(k, "")).strip() for k in sorted(r.keys()))
        r["RowId"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    client = bq_client()
    dataset = f"{BQ_PROJECT}.{BQ_DATASET}"
    target_id = f"{dataset}.{table_name}"
    ensure_table_with_schema(table_name, schema)
    staging_id = f"{dataset}.{table_name}_staging"
    client.delete_table(staging_id, not_found_ok=True)
    client.create_table(bigquery.Table(staging_id, schema=schema))
    job = client.load_table_from_json(rows, staging_id,
        job_config=bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_TRUNCATE"))
    job.result()
    merge_sql = f"""
    MERGE `{target_id}` T
    USING `{staging_id}` S
    ON T.RowId = S.RowId
    WHEN NOT MATCHED THEN
      INSERT ROW
    """
    client.query(merge_sql).result()
    client.delete_table(staging_id, not_found_ok=True)

def get_months_from_positive():
    sql = f"SELECT DISTINCT Month FROM `{BQ_PROJECT}.{BQ_DATASET}.{TARGET_TABLE}` ORDER BY Month DESC"
    return [r["Month"] for r in bq_query(sql)]

def get_month_totals(month: str | None):
    base = f"`{BQ_PROJECT}.{BQ_DATASET}.{TARGET_TABLE}`"
    where, qp = apply_filters_where({"month": month} if month else {})
    cards_sql = f"SELECT COALESCE(SUM(Amount),0) AS amt FROM {base} {where}"
    cards_total = bq_query(cards_sql, qp)[0]["amt"]
    ms_base = f"`{BQ_PROJECT}.{BQ_DATASET}.{MANUAL_SPEND_TABLE}`"
    ms_sql = f"SELECT COALESCE(SUM(Amount),0) AS amt FROM {ms_base} WHERE Month=@month" if month else \
             f"SELECT COALESCE(SUM(Amount),0) AS amt FROM {ms_base}"
    manual_total = bq_query(ms_sql, {"month": month})[0]["amt"] if month else bq_query(ms_sql)[0]["amt"]
    inc_base = f"`{BQ_PROJECT}.{BQ_DATASET}.{MONTHLY_INCOME_TABLE}`"
    inc_sql = f"SELECT COALESCE(SUM(Amount),0) AS amt FROM {inc_base} WHERE Month=@month" if month else \
              f"SELECT COALESCE(SUM(Amount),0) AS amt FROM {inc_base}"
    income_total = bq_query(inc_sql, {"month": month})[0]["amt"] if month else bq_query(inc_sql)[0]["amt"]
    return cards_total, manual_total, income_total

# Helpers: dropdowns and main totals
def get_existing_manual_categories():
    sql = f"SELECT DISTINCT Category FROM `{BQ_PROJECT}.{BQ_DATASET}.manual_spend` ORDER BY Category"
    return [r["Category"] for r in bq_query(sql)]

def get_existing_income_sources():
    sql = f"SELECT DISTINCT Source FROM `{BQ_PROJECT}.{BQ_DATASET}.monthly_income` ORDER BY Source"
    return [r["Source"] for r in bq_query(sql)]

def get_main_totals_for_month(month: str | None):
    base = f"`{BQ_PROJECT}.{BQ_DATASET}.{TARGET_TABLE}`"
    where, qp = apply_filters_where({"month": month} if month else {})
    sql = f"""
    SELECT MainCategory, SUM(Amount) AS Amount
    FROM {base}
    {where}
    GROUP BY MainCategory
    ORDER BY Amount DESC
    """
    return bq_query(sql, qp)

# ---- Combined totals for dashboard and status ----
def get_top_categories_with_manual(selected_month: str | None, where_sql: str, qp: dict):
    # Cards top categories
    table_id = f"`{BQ_PROJECT}.{BQ_DATASET}.{TARGET_TABLE}`"
    cards_sql = f"""
      SELECT Category, SUM(Amount) AS Amount
      FROM {table_id}
      {where_sql}
      GROUP BY Category
      ORDER BY Amount DESC
      LIMIT 12
    """
    top_cards = bq_query(cards_sql, qp)
    # Manual total for selected month as a single category "Manual"
    if selected_month:
        manual_base = f"`{BQ_PROJECT}.{BQ_DATASET}.manual_spend`"
        manual_sql = f"SELECT COALESCE(SUM(Amount),0) AS Amount FROM {manual_base} WHERE Month=@month"
        manual_total = float(bq_query(manual_sql, {"month": selected_month})[0]["Amount"] or 0)
        if manual_total > 0:
            top_cards.append({"Category": "Manual", "Amount": manual_total})
    return top_cards

def get_combined_monthly_totals(selected_month: str | None, trend_where_sql: str, trend_qp: dict):
    # Cards monthly totals (respect filters except Month)
    table_id = f"`{BQ_PROJECT}.{BQ_DATASET}.{TARGET_TABLE}`"
    cards_sql = f"""
      SELECT Month, SUM(Amount) AS Amount
      FROM {table_id}
      {trend_where_sql}
      GROUP BY Month
      ORDER BY Month
    """
    cards_rows = bq_query(cards_sql, trend_qp)
    # Manual monthly totals (no card filters; only Month scope)
    manual_base = f"`{BQ_PROJECT}.{BQ_DATASET}.manual_spend`"
    if selected_month:
        manual_sql = f"""
          SELECT Month, SUM(Amount) AS Amount
          FROM {manual_base}
          WHERE Month=@month
          GROUP BY Month
          ORDER BY Month
        """
        manual_rows = bq_query(manual_sql, {"month": selected_month})
    else:
        manual_sql = f"""
          SELECT Month, SUM(Amount) AS Amount
          FROM {manual_base}
          GROUP BY Month
          ORDER BY Month
        """
        manual_rows = bq_query(manual_sql)
    # Merge by Month
    m_to_total: Dict[str, float] = {}
    for r in cards_rows:
        m_to_total[str(r["Month"])] = float(r["Amount"] or 0)
    for r in manual_rows:
        m = str(r["Month"]); m_to_total[m] = m_to_total.get(m, 0.0) + float(r["Amount"] or 0)
    return [{"Month": m, "Amount": m_to_total[m]} for m in sorted(m_to_total)]

def get_year_totals_combined():
    cards_sql = f"""
      SELECT CAST(SUBSTR(Month,1,4) AS INT64) AS Year, SUM(Amount) AS Amount
      FROM `{BQ_PROJECT}.{BQ_DATASET}.{TARGET_TABLE}`
      GROUP BY Year
    """
    by_year = {int(r["Year"]): float(r["Amount"] or 0) for r in bq_query(cards_sql)}
    manual_sql = f"""
      SELECT CAST(SUBSTR(Month,1,4) AS INT64) AS Year, SUM(Amount) AS Amount
      FROM `{BQ_PROJECT}.{BQ_DATASET}.manual_spend`
      GROUP BY Year
    """
    for r in bq_query(manual_sql):
        y = int(r["Year"])
        by_year[y] = by_year.get(y, 0.0) + float(r["Amount"] or 0)
    return [{"Year": y, "Amount": by_year[y]} for y in sorted(by_year)]

def get_main_totals_all_with_manual():
    cards_sql = f"""
      SELECT MainCategory, SUM(Amount) AS Amount
      FROM `{BQ_PROJECT}.{BQ_DATASET}.{TARGET_TABLE}`
      GROUP BY MainCategory
      ORDER BY Amount DESC
    """
    rows = bq_query(cards_sql)
    manual_sql = f"SELECT COALESCE(SUM(Amount),0) AS Amount FROM `{BQ_PROJECT}.{BQ_DATASET}.manual_spend`"
    manual_total = float(bq_query(manual_sql)[0]["Amount"] or 0)
    rows.append({"MainCategory": "Manual", "Amount": manual_total})
    return rows

def get_month_stats_combined(limit: int = 12):
    cards_sql = f"""
      SELECT Month, COUNT(1) AS RowCount, SUM(Amount) AS Amount
      FROM `{BQ_PROJECT}.{BQ_DATASET}.{TARGET_TABLE}`
      GROUP BY Month
    """
    manual_sql = f"""
      SELECT Month, COUNT(1) AS RowCount, SUM(Amount) AS Amount
      FROM `{BQ_PROJECT}.{BQ_DATASET}.manual_spend`
      GROUP BY Month
    """
    combined = {str(r["Month"]): (int(r["RowCount"]), float(r["Amount"] or 0)) for r in bq_query(cards_sql)}
    for r in bq_query(manual_sql):
        m = str(r["Month"])
        rc, amt = combined.get(m, (0, 0.0))
        combined[m] = (rc + int(r["RowCount"]), amt + float(r["Amount"] or 0))
    items = [{"Month": m, "RowCount": v[0], "Amount": v[1]} for m, v in combined.items()]
    items.sort(key=lambda x: x["Month"], reverse=True)
    return items[:limit]



# ---------------- Dashboard ----------------
@app.get("/dashboard")
def dashboard():
    if not validate_dataset(BQ_PROJECT, BQ_DATASET):
        return f"Error: Dataset {BQ_PROJECT}.{BQ_DATASET} not accessible.", 500

    loaded = request.args.get("loaded", "")
    include_manual = (request.args.get("include_manual", "1").strip() == "1")  # default ON

    months, cards, mains, cats = get_distinct_filters()

    selected_month = request.args.get("month", "").strip() or None
    selected_card = request.args.get("card", "").strip() or None
    selected_main = request.args.get("main", "").strip() or None
    selected_cat = request.args.get("cat", "").strip() or None

    filter_params = {"month": selected_month, "card": selected_card, "main": selected_main, "cat": selected_cat}
    where_sql, qp = apply_filters_where(filter_params)

    table_id = f"`{BQ_PROJECT}.{BQ_DATASET}.{TARGET_TABLE}`"
    meta = get_table_metadata(TARGET_TABLE)

    latest_month_sql = f"SELECT Month FROM {table_id} WHERE Month IS NOT NULL ORDER BY Month DESC LIMIT 1"
    latest_month_rows = bq_query(latest_month_sql)
    if not latest_month_rows:
        return render_template(
            "dashboard.html",
            latest_month="",
            month_for_view="",
            top_categories=[],
            monthly_totals=[],
            latest_details=[],
            project=BQ_PROJECT,
            dataset=BQ_DATASET,
            aggregate_labels=[],
            aggregate_values=[],
            aggregate_group_by="",
            aggregate_month="",
            aggregate_min_amount="",
            aggregate_rows=[],
            months=months, cards=cards, mains=mains, cats=cats,
            selected_month=selected_month, selected_card=selected_card,
            selected_main=selected_main, selected_cat=selected_cat,
            include_manual=include_manual,
            loaded=loaded,
            table_modified=meta.get("modified"),
        )

    latest_month = latest_month_rows[0]["Month"]
    month_for_view = selected_month or latest_month

    # Top categories with Manual category appended when a month is selected
    top_categories = get_top_categories_with_manual(selected_month, where_sql, qp)

    # Monthly totals (cards + manual)
    trend_where_params = {k: v for k, v in filter_params.items() if k != "month"}
    trend_where_sql, trend_qp = apply_filters_where(trend_where_params)
    monthly_totals = get_combined_monthly_totals(selected_month, trend_where_sql, trend_qp)

    # Unified Details (cards + manual normalized)
    latest_details = get_dashboard_details_with_manual(selected_month, where_sql, qp, include_manual)

    return render_template(
        "dashboard.html",
        latest_month=latest_month,
        month_for_view=month_for_view,
        top_categories=top_categories,
        monthly_totals=monthly_totals,
        latest_details=latest_details,
        project=BQ_PROJECT,
        dataset=BQ_DATASET,
        aggregate_labels=[],
        aggregate_values=[],
        aggregate_group_by="",
        aggregate_month="",
        aggregate_min_amount="",
        aggregate_rows=[],
        months=months, cards=cards, mains=mains, cats=cats,
        selected_month=selected_month, selected_card=selected_card,
        selected_main=selected_main, selected_cat=selected_cat,
        include_manual=include_manual,
        loaded=loaded,
        table_modified=meta.get("modified"),
    )


def normalize_group_by_param(vals: list[str]) -> list[str]:
    allowed = {"Month", "CardName", "MainCategory", "Category"}
    if not vals:
        return []
    return [v for v in vals if v in allowed]

@app.get("/aggregate")
def aggregate():
    if not validate_dataset(BQ_PROJECT, BQ_DATASET):
        return f"Error: Dataset {BQ_PROJECT}.{BQ_DATASET} not accessible.", 500
    table_id = f"`{BQ_PROJECT}.{BQ_DATASET}.{TARGET_TABLE}`"
    group_by_list = request.args.getlist("group_by")
    group_cols = normalize_group_by_param(group_by_list) or ["Category"]
    selected_month = request.args.get("month", "").strip() or None
    selected_card = request.args.get("card", "").strip() or None
    selected_main = request.args.get("main", "").strip() or None
    selected_cat = request.args.get("cat", "").strip() or None
    min_amount = request.args.get("min_amount", "").strip()
    try:
        min_amount_val = float(min_amount) if min_amount else None
    except Exception:
        min_amount_val = None
    months, cards, mains, cats = get_distinct_filters()
    filter_params = {"month": selected_month, "card": selected_card, "main": selected_main, "cat": selected_cat}
    where_sql, qp = apply_filters_where(filter_params)
    select_cols = ", ".join(group_cols)
    group_cols_sql = ", ".join(group_cols)
    order_clause = "ORDER BY Month ASC, Amount DESC" if "Month" in group_cols else "ORDER BY Amount DESC"
    sql = f"""
      SELECT {select_cols}, SUM(Amount) AS Amount
      FROM {table_id}
      {where_sql}
      GROUP BY {group_cols_sql}
      {order_clause}
      LIMIT 1000
    """
    rows = bq_query(sql, params=qp)
    if min_amount_val is not None:
        rows = [r for r in rows if (r.get("Amount") or 0) >= min_amount_val]
    def label_for_row(r):
        return " / ".join(str(r.get(c, "")) for c in group_cols)
    labels = [label_for_row(r) for r in rows]
    values = [float(r.get("Amount") or 0) for r in rows]
    return render_template(
        "dashboard.html",
        latest_month="",
        month_for_view=selected_month or "",
        top_categories=[],
        monthly_totals=[],
        latest_details=[],
        project=BQ_PROJECT,
        dataset=BQ_DATASET,
        aggregate_labels=labels,
        aggregate_values=values,
        aggregate_group_by=",".join(group_cols),
        aggregate_month=selected_month or "",
        aggregate_min_amount=min_amount or "",
        aggregate_rows=rows,
        months=months, cards=cards, mains=mains, cats=cats,
        selected_month=selected_month, selected_card=selected_card,
        selected_main=selected_main, selected_cat=selected_cat,
        loaded="",
        table_modified=get_table_metadata(TARGET_TABLE).get("modified"),
    )

# ---------------- Status (combined) ----------------
@app.get("/status")
def status():
    if not validate_dataset(BQ_PROJECT, BQ_DATASET):
        return f"Error: Dataset {BQ_PROJECT}.{BQ_DATASET} not accessible.", 500
    target = TARGET_TABLE
    meta = get_table_metadata(target)
    month_stats = get_month_stats_combined(limit=12)
    year_totals = get_year_totals_combined()
    total_amount_all = sum(float(r["Amount"] or 0) for r in year_totals)
    main_totals_all = get_main_totals_all_with_manual()
    return render_template(
        "status.html",
        project=BQ_PROJECT,
        dataset=BQ_DATASET,
        table=target,
        meta=meta,
        month_stats=month_stats,
        last_insert={},  # unused
        total_amount_all=total_amount_all,
        main_totals_all=main_totals_all,
        year_totals=year_totals
    )


# ---------------- Review routes ----------------
@app.get("/review")
def review_get():
    month = request.args.get("month", "").strip()
    card  = request.args.get("card", "").strip() or None
    main  = request.args.get("main", "").strip() or None
    cat   = request.args.get("cat", "").strip() or None
    tab   = request.args.get("tab", "").strip()
    months, cards, mains, cats = get_distinct_filters()
    table_id = f"`{BQ_PROJECT}.{BQ_DATASET}.{TARGET_TABLE}`"
    where = []; qp = {}
    if month: where.append("Month = @month"); qp["month"] = month
    if card:  where.append("CardName = @card"); qp["card"] = card
    if main:  where.append("MainCategory = @main"); qp["main"] = main
    if cat:   where.append("Category = @cat"); qp["cat"] = cat
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    sql = f"""
      SELECT Month, CardName, MainCategory, Category, Description, Amount, Comment, RowHash
      FROM {table_id}
      {where_sql}
      ORDER BY Month DESC, Amount DESC
      LIMIT 500
    """
    rows = bq_query(sql, params=qp)
    msg = request.args.get("msg", "").strip()
    msg_type = request.args.get("msg_type", "").strip()
    return render_template(
        "review.html",
        project=BQ_PROJECT, dataset=BQ_DATASET,
        months=months, cards=cards, mains=mains, cats=cats,
        selected_month=month or "", selected_card=card, selected_main=main, selected_cat=cat,
        review_rows=rows,
        message=msg, message_type=msg_type,
        tab=tab
    )

@app.post("/review")
def review_post():
    rowhash    = request.form.get("rowhash", "").strip()
    percentage = request.form.get("percentage", "").strip()
    note       = request.form.get("note", "").strip()
    month = (request.form.get("month") or "").strip()
    card  = (request.form.get("card") or "").strip()
    main  = (request.form.get("main") or "").strip()
    cat   = (request.form.get("cat") or "").strip()
    tab   = (request.form.get("tab") or "").strip() or "review"
    if not rowhash or not percentage:
        return redirect(url_for("review_get",
                                month=month, card=card, main=main, cat=cat, tab=tab,
                                msg="rowhash and percentage are required", msg_type="error"), code=303)
    try:
        pct = float(percentage)
        if pct < 0 or pct > 100:
            return redirect(url_for("review_get",
                                    month=month, card=card, main=main, cat=cat, tab=tab,
                                    msg="percentage must be between 0 and 100", msg_type="error"), code=303)
    except ValueError:
        return redirect(url_for("review_get",
                                month=month, card=card, main=main, cat=cat, tab=tab,
                                msg="percentage must be numeric", msg_type="error"), code=303)
    table_id = f"`{BQ_PROJECT}.{BQ_DATASET}.{TARGET_TABLE}`"
    sel_sql = f"""
      SELECT Amount, Comment
      FROM {table_id}
      WHERE RowHash = @rowhash
      LIMIT 1
    """
    recs = bq_query(sel_sql, params={"rowhash": rowhash})
    if not recs:
        return redirect(url_for("review_get",
                                month=month, card=card, main=main, cat=cat, tab=tab,
                                msg="Row not found", msg_type="error"), code=303)
    original_amount = float(recs[0].get("Amount") or 0.0)
    existing_comment = (recs[0].get("Comment") or "").strip()
    if "[REVIEW]" in existing_comment:
        return redirect(url_for("review_get",
                                month=month, card=card, main=main, cat=cat, tab=tab,
                                msg="This row was already reviewed. No changes applied.", msg_type="error"), code=303)
    share_amount = round(original_amount * (pct / 100.0), 2)
    fmt_orig = f"${original_amount:,.2f}"
    fmt_share = f"${share_amount:,.2f}"
    structured = f"[REVIEW] Original: {fmt_orig}, Share: {pct:.2f}% -> {fmt_share}"
    final_comment = structured if not existing_comment else f"{existing_comment} | {structured}"
    if note:
        final_comment = f"{final_comment}. Note: {escape(note)}"
    upd_sql = f"""
      UPDATE {table_id}
      SET Amount = @new_amount,
          Comment = @comment
      WHERE RowHash = @rowhash
    """
    client = bq_client()
    job = client.query(
        upd_sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("new_amount", "FLOAT", share_amount),
                bigquery.ScalarQueryParameter("comment", "STRING", final_comment),
                bigquery.ScalarQueryParameter("rowhash", "STRING", rowhash),
            ]
        )
    )
    job.result()
    msg = f"Adjusted to {fmt_share} ({pct:.2f}%). Review note saved."
    return redirect(url_for("review_get",
                            month=month, card=card, main=main, cat=cat, tab=tab,
                            msg=msg, msg_type="ok"), code=303)


# ---- Include manual spend in Dashboard details ----
def get_dashboard_details_with_manual(selected_month: str | None, where_sql: str, qp: dict, include_manual: bool):
    """
    Returns a unified list of detail rows. Card rows honor all filters. Manual rows honor Month only
    and are normalized to the card schema with CardName='Manual', MainCategory='Manual'.
    """
    table_id = f"`{BQ_PROJECT}.{BQ_DATASET}.{TARGET_TABLE}`"

    # Card rows
    cards_sql = f"""
      SELECT CardName, MainCategory, Category, Description, Amount, Month
      FROM {table_id}
      {where_sql}
      ORDER BY Month DESC, Amount DESC
      LIMIT 1000
    """
    cards_rows = bq_query(cards_sql, qp)

    # Manual rows (Month scope)
    manual_rows = []
    if include_manual:
        manual_base = f"`{BQ_PROJECT}.{BQ_DATASET}.manual_spend`"
        if selected_month:
            manual_sql = f"""
              SELECT Month, Category AS MainCategory, Category, Description, Amount
              FROM {manual_base}
              WHERE Month=@month
              ORDER BY Amount DESC
              LIMIT 1000
            """
            mrows = bq_query(manual_sql, {"month": selected_month})
        else:
            manual_sql = f"""
              SELECT Month, Category AS MainCategory, Category, Description, Amount
              FROM {manual_base}
              ORDER BY Month DESC, Amount DESC
              LIMIT 1000
            """
            mrows = bq_query(manual_sql)

        manual_rows = [{
            "CardName": "Manual",
            "MainCategory": "Manual",
            "Category": r.get("Category"),
            "Description": r.get("Description"),
            "Amount": r.get("Amount"),
            "Month": r.get("Month"),
        } for r in mrows]

    return cards_rows + manual_rows


# ---------------- Bills helpers and routes ----------------
BILLS_TABLE = "credit_card_bills"
MASTER_MONTH = "MASTER"

def ensure_bills_table():
    client = bq_client()
    table_id = f"{BQ_PROJECT}.{BQ_DATASET}.{BILLS_TABLE}"
    schema = [
        bigquery.SchemaField("CardName", "STRING"),
        bigquery.SchemaField("DueDay", "INTEGER"),
        bigquery.SchemaField("BillMonth", "STRING"),
        bigquery.SchemaField("Amount", "FLOAT"),
        bigquery.SchemaField("Paid", "BOOL"),
        bigquery.SchemaField("PaidAt", "TIMESTAMP"),
        bigquery.SchemaField("Note", "STRING"),
        bigquery.SchemaField("RowId", "STRING"),
    ]
    try:
        client.get_table(table_id)
    except Exception:
        client.create_table(bigquery.Table(table_id, schema=schema))

def month_str(dt=None):
    import datetime as _dt
    dt = dt or _dt.date.today()
    return dt.strftime("%Y-%m")

def get_card_masters():
    table_id = f"`{BQ_PROJECT}.{BQ_DATASET}.{BILLS_TABLE}`"
    sql = f"""
      SELECT CardName, ANY_VALUE(DueDay) AS DueDay
      FROM {table_id}
      WHERE BillMonth = @m
      GROUP BY CardName
      ORDER BY CardName
    """
    return bq_query(sql, {"m": MASTER_MONTH})

@app.get("/bills")
def bills():
    if not validate_dataset(BQ_PROJECT, BQ_DATASET):
        return f"Error: Dataset {BQ_PROJECT}.{BQ_DATASET} not accessible.", 500
    ensure_bills_table()
    import datetime as dt
    qs_month = request.args.get("m", "").strip()
    unpaid_only = request.args.get("unpaid", "").strip() == "1"
    bill_month = qs_month if qs_month else month_str(dt.date.today())
    table_id = f"`{BQ_PROJECT}.{BQ_DATASET}.{BILLS_TABLE}`"
    summary_sql = f"""
      SELECT
        COALESCE(SUM(Amount), 0) AS TotalAmount,
        SUM(CASE WHEN Paid THEN 1 ELSE 0 END) AS PaidCount,
        SUM(CASE WHEN NOT Paid THEN 1 ELSE 0 END) AS UnpaidCount
      FROM {table_id}
      WHERE BillMonth = @m
      {"AND NOT Paid" if unpaid_only else ""}
    """
    summary = bq_query(summary_sql, {"m": bill_month})
    summary = summary[0] if summary else {"TotalAmount": 0, "PaidCount": 0, "UnpaidCount": 0}
    rows_sql = f"""
      SELECT CardName, DueDay, BillMonth, Amount, Paid, PaidAt, Note, RowId
      FROM {table_id}
      WHERE BillMonth = @m
      {"AND NOT Paid" if unpaid_only else ""}
      ORDER BY CASE WHEN Paid THEN 1 ELSE 0 END ASC, DueDay ASC, CardName ASC
    """
    bills_rows = bq_query(rows_sql, {"m": bill_month})
    cards = get_card_masters()
    try:
        y, mo = map(int, bill_month.split("-"))
        cur = dt.date(y, mo, 1)
        prev_m = (cur.replace(day=1) - dt.timedelta(days=1)).strftime("%Y-%m")
        next_m = (cur + dt.timedelta(days=32)).replace(day=1).strftime("%Y-%m")
    except Exception:
        prev_m, next_m = "", ""
    return render_template(
        "bills.html",
        project=BQ_PROJECT, dataset=BQ_DATASET,
        bill_month=bill_month,
        summary=summary,
        bills=bills_rows,
        cards=cards,
        unpaid_only=unpaid_only,
        prev_month=prev_m, next_month=next_m
    )

@app.post("/bills/add_card")
def bills_add_card():
    if not validate_dataset(BQ_PROJECT, BQ_DATASET):
        return f"Error: Dataset {BQ_PROJECT}.{BQ_DATASET} not accessible.", 500
    ensure_bills_table()
    card = (request.form.get("card") or "").strip()
    due_day = (request.form.get("due_day") or "").strip()
    note = (request.form.get("note") or "").strip()
    view_month = (request.form.get("view_month") or "").strip() or month_str()
    if not card or not due_day:
        return redirect(url_for("bills", m=view_month), code=303)
    try:
        due_day_int = int(due_day)
    except Exception:
        return redirect(url_for("bills", m=view_month), code=303)
    row_id = hashlib.sha256(f"{card}|{MASTER_MONTH}".encode("utf-8")).hexdigest()[:16]
    table_id = f"`{BQ_PROJECT}.{BQ_DATASET}.{BILLS_TABLE}`"
    up_sql = f"""
      MERGE {table_id} T
      USING (SELECT @CardName AS CardName, @BillMonth AS BillMonth) S
      ON T.CardName = S.CardName AND T.BillMonth = S.BillMonth
      WHEN MATCHED THEN
        UPDATE SET DueDay=@DueDay, Note=@Note
      WHEN NOT MATCHED THEN
        INSERT (CardName, DueDay, BillMonth, Amount, Paid, PaidAt, Note, RowId)
        VALUES (@CardName, @DueDay, @BillMonth, 0.0, FALSE, NULL, @Note, @RowId)
    """
    client = bq_client()
    job = client.query(
        up_sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("CardName", "STRING", card),
                bigquery.ScalarQueryParameter("BillMonth", "STRING", MASTER_MONTH),
                bigquery.ScalarQueryParameter("DueDay", "INT64", due_day_int),
                bigquery.ScalarQueryParameter("Note", "STRING", note or "CARD_MASTER"),
                bigquery.ScalarQueryParameter("RowId", "STRING", row_id),
            ]
        )
    ); job.result()
    return redirect(url_for("bills", m=view_month), code=303)

@app.post("/bills/add")
def bills_add():
    if not validate_dataset(BQ_PROJECT, BQ_DATASET):
        return f"Error: Dataset {BQ_PROJECT}.{BQ_DATASET} not accessible.", 500
    ensure_bills_table()
    card = (request.form.get("card") or "").strip()
    amount = (request.form.get("amount") or "").strip()
    note = (request.form.get("note") or "").strip()
    bill_month = (request.form.get("bill_month") or "").strip() or month_str()
    if not card or not amount:
        return redirect(url_for("bills", m=bill_month), code=303)
    try:
        amt = float(amount)
    except Exception:
        return redirect(url_for("bills", m=bill_month), code=303)
    table_id = f"`{BQ_PROJECT}.{BQ_DATASET}.{BILLS_TABLE}`"
    dd_sql = f"""
      SELECT ANY_VALUE(DueDay) AS DueDay
      FROM {table_id}
      WHERE BillMonth = @master AND CardName = @card
      LIMIT 1
    """
    dd_rows = bq_query(dd_sql, {"master": MASTER_MONTH, "card": card})
    if not dd_rows or dd_rows[0].get("DueDay") is None:
        return redirect(url_for("bills", m=bill_month), code=303)
    due_day_int = int(dd_rows[0]["DueDay"])
    row_id = hashlib.sha256(f"{card}|{bill_month}".encode("utf-8")).hexdigest()[:16]
    upd_sql = f"""
      MERGE {table_id} T
      USING (SELECT @CardName AS CardName, @BillMonth AS BillMonth) S
      ON T.CardName = S.CardName AND T.BillMonth = S.BillMonth
      WHEN MATCHED THEN
        UPDATE SET Amount=@Amount, Note=@Note
      WHEN NOT MATCHED THEN
        INSERT (CardName, DueDay, BillMonth, Amount, Paid, PaidAt, Note, RowId)
        VALUES (@CardName, @DueDay, @BillMonth, @Amount, FALSE, NULL, @Note, @RowId)
    """
    client = bq_client()
    job = client.query(
        upd_sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("CardName", "STRING", card),
                bigquery.ScalarQueryParameter("BillMonth", "STRING", bill_month),
                bigquery.ScalarQueryParameter("DueDay", "INT64", due_day_int),
                bigquery.ScalarQueryParameter("Amount", "FLOAT", amt),
                bigquery.ScalarQueryParameter("Note", "STRING", note),
                bigquery.ScalarQueryParameter("RowId", "STRING", row_id),
            ]
        )
    ); job.result()
    return redirect(url_for("bills", m=bill_month), code=303)

@app.post("/bills/mark_paid")
def bills_mark_paid():
    if not validate_dataset(BQ_PROJECT, BQ_DATASET):
        return f"Error: Dataset {BQ_PROJECT}.{BQ_DATASET} not accessible.", 500
    ensure_bills_table()
    row_id = (request.form.get("row_id") or "").strip()
    bill_month = (request.form.get("bill_month") or "").strip() or month_str()
    if not row_id:
        return redirect(url_for("bills", m=bill_month), code=303)
    table_id = f"`{BQ_PROJECT}.{BQ_DATASET}.{BILLS_TABLE}`"
    upd_sql = f"""
      UPDATE {table_id}
      SET Paid = TRUE, PaidAt = CURRENT_TIMESTAMP()
      WHERE RowId = @RowId
    """
    client = bq_client()
    job = client.query(
        upd_sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("RowId", "STRING", row_id)]
        )
    ); job.result()
    return redirect(url_for("bills", m=bill_month), code=303)

# ---- Categories page ----
@app.get("/categories")
def categories_get():
    desc = request.args.get("desc", "").strip()
    msg = request.args.get("msg", "").strip()
    msg_type = request.args.get("msg_type", "").strip()
    regex_index = build_regex_index_bq()
    cat, main = classify_with_bq_index(desc, regex_index) if desc else ("", "")
    rows = fetch_categories_bq()
    mains = sorted({r[0] for r in rows})
    categories_by_main = fetch_categories_by_main()
    other_rows = fetch_other_monthly(limit=200)
    return render_template(
        "categories.html",
        description=desc,
        result_cat=cat,
        result_main=main,
        mains=mains,
        categories_by_main=categories_by_main,
        message=msg,
        message_type=msg_type,
        other_rows=other_rows
    )

@app.post("/categories/add")
def categories_add_post():
    desc = request.form.get("desc", "").strip()
    main = (request.form.get("main", "") or "Misc").strip()
    category = request.form.get("category", "").strip()
    keywords_raw = request.form.get("keywords", "").strip()
    if not category or not keywords_raw:
        return redirect(url_for("categories_get", desc=desc, msg="Category and keywords are required", msg_type="error"))
    keywords = [k.strip().lower() for k in keywords_raw.split(",") if k.strip()]
    insert_keywords_bq(main, category, keywords)
    table_id = f"{BQ_PROJECT}.{BQ_DATASET}.{TARGET_TABLE}"
    params = [
        bigquery.ScalarQueryParameter("category", "STRING", category),
        bigquery.ScalarQueryParameter("main", "STRING", main),
    ]
    cond_parts = []
    for i, kw in enumerate(keywords):
        use_boundaries = bool(re.fullmatch(r'[\w\s]+', kw))
        pattern = (r"\b" + re.escape(kw) + r"\b") if use_boundaries else re.escape(kw)
        pname = f"pat{i}"
        params.append(bigquery.ScalarQueryParameter(pname, "STRING", pattern))
        cond_parts.append(f"REGEXP_CONTAINS(LOWER(Description), @{pname})")
    cond_sql = " OR ".join(cond_parts) if cond_parts else "FALSE"
    upd_sql = f"""
    UPDATE `{table_id}`
    SET Category = @category, MainCategory = @main
    WHERE LOWER(IFNULL(Category, 'other')) = 'other'
      AND ({cond_sql})
    """
    client = bq_client()
    job = client.query(upd_sql, job_config=bigquery.QueryJobConfig(query_parameters=params)); job.result()
    return redirect(url_for("categories_get", desc=desc, msg=f"Added '{category}' under '{main}' and updated matching rows.", msg_type="ok"))

@app.post("/categories/reclassify_monthly")
def categories_reclassify_monthly():
    month = (request.form.get("month", "") or "").strip()
    card = (request.form.get("card", "") or "").strip()
    desc = (request.form.get("desc", "") or "").strip()
    main = (request.form.get("main", "") or "").strip()
    category = (request.form.get("category", "") or "").strip()
    if not (month and card and desc and main and category):
        return redirect(url_for("categories_get", msg="All fields required to reclassify this row.", msg_type="error"))
    table_id = f"{BQ_PROJECT}.{BQ_DATASET}.{TARGET_TABLE}"
    upd_sql = f"""
    UPDATE `{table_id}`
    SET MainCategory = @main, Category = @category
    WHERE Month = @month
      AND CardName = @card
      AND LOWER(TRIM(Description)) = LOWER(TRIM(@desc))
    """
    client = bq_client()
    job = client.query(
        upd_sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("main", "STRING", main),
                bigquery.ScalarQueryParameter("category", "STRING", category),
                bigquery.ScalarQueryParameter("month", "STRING", month),
                bigquery.ScalarQueryParameter("card", "STRING", card),
                bigquery.ScalarQueryParameter("desc", "STRING", desc),
            ]
        )
    ); job.result()
    return redirect(url_for("categories_get", msg="Row reclassified.", msg_type="ok"))

@app.post("/upload")
def upload():
    try:
        files = request.files.getlist("files")
        if not files:
            return "No files uploaded.", 400
        frames = []
        for f in files:
            b = f.read()
            df = fetch_csv_to_df_bytes(b)
            derived_card = derive_card_name_from_path(f.filename)
            df_clean = clean_and_standardize(df, card_name=derived_card)
            frames.append(df_clean)
        classifier = make_classifier_grouped("/dev/null")
        result = run_pipeline(frames, classifier)
        records = result["all_positive_monthly"]
        new_rows = append_non_duplicates(records, TARGET_TABLE)
        latest_month = str(result.get("latest_month"))
        return redirect(url_for("dashboard", month=latest_month, loaded=new_rows), code=303)
    except Exception as e:
        return f"Upload error: {type(e).__name__}: {e}", 500

# ---------------- Budget & Income ----------------
@app.get("/budget")
def budget_get():
    month = request.args.get("month", "").strip() or None
    months = get_months_from_positive()
    manual_categories = get_existing_manual_categories()
    income_sources = get_existing_income_sources()
    ms_sql = f"""
      SELECT Month, Category, Description, Amount, Note
      FROM `{BQ_PROJECT}.{BQ_DATASET}.manual_spend`
      { 'WHERE Month=@month' if month else '' }
      ORDER BY Amount DESC
    """
    manual_rows = bq_query(ms_sql, {"month": month} if month else None)
    inc_sql = f"""
      SELECT Month, Source, Amount, Note
      FROM `{BQ_PROJECT}.{BQ_DATASET}.monthly_income`
      { 'WHERE Month=@month' if month else '' }
      ORDER BY Amount DESC
    """
    income_rows = bq_query(inc_sql, {"month": month} if month else None)
    cards_total, manual_total, income_total = get_month_totals(month)
    status_txt = "Within limit" if (cards_total + manual_total) <= income_total else "Exceeded"
    main_totals = get_main_totals_for_month(month)
    return render_template(
        "budget.html",
        months=months,
        selected_month=month or "",
        manual_rows=manual_rows,
        income_rows=income_rows,
        cards_total=cards_total,
        manual_total=manual_total,
        income_total=income_total,
        status=status_txt,
        main_totals=main_totals,
        manual_categories=manual_categories,
        income_sources=income_sources
    )

@app.post("/budget/spend/add")
def budget_spend_add():
    month = request.form.get("month", "").strip()
    category = (request.form.get("category_new") or request.form.get("category") or "").strip()
    desc = request.form.get("description", "").strip()
    amount = float(request.form.get("amount", "0") or 0)
    note = request.form.get("note", "").strip()
    if not month or amount <= 0 or not desc or not category:
        return redirect(url_for("budget_get", month=month))
    upsert_simple(MANUAL_SPEND_TABLE, MANUAL_SPEND_SCHEMA, [{
        "Month": month, "Category": category, "Description": desc, "Amount": amount, "Note": note
    }])
    return redirect(url_for("budget_get", month=month))

@app.post("/budget/income/add")
def budget_income_add():
    month = request.form.get("month", "").strip()
    source = (request.form.get("source_new") or request.form.get("source") or "").strip()
    amount = float(request.form.get("amount", "0") or 0)
    note = request.form.get("note", "").strip()
    if not month or amount <= 0 or not source:
        return redirect(url_for("budget_get", month=month))
    upsert_simple(MONTHLY_INCOME_TABLE, MONTHLY_INCOME_SCHEMA, [{
        "Month": month, "Source": source, "Amount": amount, "Note": note
    }])
    return redirect(url_for("budget_get", month=month))

# ---------------- Entrypoint ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
