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
    qjob = client.query(dedup_sql)
    qjob.result()

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

# ---- Category config helpers ----

CATEGORIES_JSON_PATH = Path("categories_grouped.json")

def read_categories() -> List[Dict[str, Any]]:
    with CATEGORIES_JSON_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_categories(cfg: List[Dict[str, Any]]):
    # simple pretty print, keep structure consistent
    with CATEGORIES_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

def add_category(cfg: List[Dict[str, Any]], main: str, category: str, keywords: List[str]) -> List[Dict[str, Any]]:
    # find main block
    for block in cfg:
        if block.get("main") == main:
            block.setdefault("categories", [])
            # if already present, merge keywords (dedup, case-insensitive)
            for entry in block["categories"]:
                if entry.get("category") == category:
                    existing = set(k.strip().lower() for k in entry.get("keywords", []))
                    for k in keywords:
                        ek = k.strip().lower()
                        if ek and ek not in existing:
                            existing.add(ek)
                    entry["keywords"] = sorted(existing)
                    return cfg
            # new entry
            block["categories"].append({"category": category, "keywords": keywords})
            return cfg
    # if main not found, create it
    cfg.append({
        "main": main,
        "categories": [
            {"category": category, "keywords": keywords}
        ]
    })
    return cfg

# build regex index exactly like your build_regex_index_grouped, but callable here
def build_regex_index_for_runtime(cfg: List[Dict[str, Any]], use_word_boundaries: bool = True) -> List[Tuple[re.Pattern, str, str]]:
    idx = []
    for block in cfg:
        main = block.get("main")
        for entry in block.get("categories", []):
            cat = entry.get("category")
            for kw in entry.get("keywords", []):
                base = kw
                if use_word_boundaries:
                    pattern = re.compile(rf"\b{re.escape(base)}\b", re.IGNORECASE)
                else:
                    pattern = re.compile(re.escape(base), re.IGNORECASE)
                idx.append((pattern, cat, main))
    return idx

def classify_with_index(text: str, regex_index: List[Tuple[re.Pattern, str, str]]) -> Tuple[str, str]:
    t = (text or "").strip()
    for pattern, cat, main in regex_index:
        if pattern.search(t):
            return cat, main
    return "Other", "Other"


# ---------------- Routes: index/list/process ----------------
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

# ---------------- Dashboard + Aggregate + Status ----------------
@app.get("/dashboard")
def dashboard():
    if not validate_dataset(BQ_PROJECT, BQ_DATASET):
        return f"Error: Dataset {BQ_PROJECT}.{BQ_DATASET} not accessible.", 500

    loaded = request.args.get("loaded", "")
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
            loaded=loaded,
            table_modified=meta.get("modified"),
        )

    latest_month = latest_month_rows[0]["Month"]
    month_for_view = selected_month or latest_month

    top_categories_sql = f"""
      SELECT Category, SUM(Amount) AS Amount
      FROM {table_id}
      {where_sql}
      GROUP BY Category
      ORDER BY Amount DESC
      LIMIT 12
    """
    top_categories = bq_query(top_categories_sql, params=qp)

    trend_where_params = {k: v for k, v in filter_params.items() if k != "month"}
    trend_where_sql, trend_qp = apply_filters_where(trend_where_params)
    monthly_totals_sql = f"""
      SELECT Month, SUM(Amount) AS Amount
      FROM {table_id}
      {trend_where_sql}
      GROUP BY Month
      ORDER BY Month
    """
    monthly_totals = bq_query(monthly_totals_sql, params=trend_qp)

    latest_detail_sql = f"""
      SELECT CardName, MainCategory, Category, Description, Amount
      FROM {table_id}
      {where_sql}
      ORDER BY Month DESC, Amount DESC
      LIMIT 1000
    """
    latest_details = bq_query(latest_detail_sql, params=qp)

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

    if "Month" in group_cols:
        order_clause = "ORDER BY Month ASC, Amount DESC"
    else:
        order_clause = "ORDER BY Amount DESC"

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

@app.get("/status")
def status():
    if not validate_dataset(BQ_PROJECT, BQ_DATASET):
        return f"Error: Dataset {BQ_PROJECT}.{BQ_DATASET} not accessible.", 500

    target = TARGET_TABLE
    meta = get_table_metadata(target)
    table_id = f"`{BQ_PROJECT}.{BQ_DATASET}.{target}`"
    by_month_sql = f"""
      SELECT Month, COUNT(*) AS RowCount, SUM(Amount) AS Amount
      FROM {table_id}
      GROUP BY Month
      ORDER BY Month DESC
      LIMIT 12
    """
    month_stats = bq_query(by_month_sql)
    last_insert_sql = f"""
      SELECT Month, MAX(Amount) AS MaxAmount
      FROM {table_id}
      GROUP BY Month
      ORDER BY Month DESC
      LIMIT 1
    """
    last_insert_rows = bq_query(last_insert_sql)
    last_insert = last_insert_rows[0] if last_insert_rows else {}
    return render_template(
        "status.html",
        project=BQ_PROJECT,
        dataset=BQ_DATASET,
        table=target,
        meta=meta,
        month_stats=month_stats,
        last_insert=last_insert,
    )

# ---------------- Review routes ----------------
@app.get("/review")
def review_get():
    month = request.args.get("month", "").strip()
    card = request.args.get("card", "").strip() or None
    main = request.args.get("main", "").strip() or None
    cat  = request.args.get("cat", "").strip() or None

    months, cards, mains, cats = get_distinct_filters()
    table_id = f"`{BQ_PROJECT}.{BQ_DATASET}.{TARGET_TABLE}`"

    rows = []
    if month:
        where = ["Month = @month"]
        qp = {"month": month}
        if card: where.append("CardName = @card"); qp["card"] = card
        if main: where.append("MainCategory = @main"); qp["main"] = main
        if cat:  where.append("Category = @cat"); qp["cat"] = cat
        where_sql = "WHERE " + " AND ".join(where)
        sql = f"""
          SELECT Month, CardName, MainCategory, Category, Description, Amount, Comment, RowHash
          FROM {table_id}
          {where_sql}
          ORDER BY Amount DESC
          LIMIT 500
        """
        rows = bq_query(sql, params=qp)

    msg = request.args.get("msg", "").strip()
    msg_type = request.args.get("msg_type", "").strip()
    return render_template(
        "review.html",
        project=BQ_PROJECT, dataset=BQ_DATASET,
        months=months, cards=cards, mains=mains, cats=cats,
        selected_month=month, selected_card=card, selected_main=main, selected_cat=cat,
        review_rows=rows,
        message=msg, message_type=msg_type
    )

@app.post("/review")
def review_post():
    rowhash    = request.form.get("rowhash", "").strip()
    percentage = request.form.get("percentage", "").strip()
    note       = request.form.get("note", "").strip()
    month      = request.form.get("month", "").strip()

    if not rowhash or not percentage:
        return redirect(url_for("review_get", month=month, msg="rowhash and percentage are required", msg_type="error"), code=303)

    try:
        pct = float(percentage)
        if pct < 0 or pct > 100:
            return redirect(url_for("review_get", month=month, msg="percentage must be between 0 and 100", msg_type="error"), code=303)
    except ValueError:
        return redirect(url_for("review_get", month=month, msg="percentage must be numeric", msg_type="error"), code=303)

    table_id = f"`{BQ_PROJECT}.{BQ_DATASET}.{TARGET_TABLE}`"
    sel_sql = f"""
      SELECT Amount, Comment
      FROM {table_id}
      WHERE RowHash = @rowhash
      LIMIT 1
    """
    recs = bq_query(sel_sql, params={"rowhash": rowhash})
    if not recs:
        return redirect(url_for("review_get", month=month, msg="Row not found", msg_type="error"), code=303)

    original_amount = float(recs[0].get("Amount") or 0.0)
    existing_comment = (recs[0].get("Comment") or "").strip()

    if "[REVIEW]" in existing_comment:
        return redirect(url_for("review_get", month=month, msg="This row was already reviewed. No changes applied.", msg_type="error"), code=303)

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
    return redirect(url_for("review_get", month=month, msg=msg, msg_type="ok"), code=303)

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

    rows_sql = f""" SELECT CardName, DueDay, BillMonth, Amount, Paid, PaidAt, Note, RowId 
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
    )
    job.result()
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
    )
    job.result()
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
    )
    job.result()
    return redirect(url_for("bills", m=bill_month), code=303)


# ---- Category test & add page ----

@app.get("/categories")
def categories_get():
    desc = request.args.get("desc", "").strip()
    cfg = read_categories()
    regex_index = build_regex_index_for_runtime(cfg)
    cat, main = classify_with_index(desc, regex_index) if desc else ("", "")
    # provide main options for select
    mains = [block.get("main") for block in cfg]
    return render_template(
        "categories.html",
        description=desc,
        result_cat=cat,
        result_main=main,
        mains=mains
    )

@app.post("/categories/add")
def categories_add_post():
    # form fields
    desc = request.form.get("desc", "").strip()
    main = request.form.get("main", "").strip() or "Misc"
    category = request.form.get("category", "").strip()
    keywords_raw = request.form.get("keywords", "").strip()
    # basic validation
    if not category or not keywords_raw:
        return redirect(url_for("categories_get", desc=desc, msg="Category and keywords are required", msg_type="error"))
    # parse keywords: comma separated
    keywords = [k.strip().lower() for k in keywords_raw.split(",") if k.strip()]
    # write JSON
    cfg = read_categories()
    cfg = add_category(cfg, main, category, keywords)
    write_categories(cfg)

    # rebuild index
    regex_index = build_regex_index_for_runtime(cfg)

    # BigQuery: update existing "Other" rows that match these new keywords
    # NOTE: keep consistent with your dataset/table names and columns
    bq_project = BQ_PROJECT
    bq_dataset = BQ_DATASET
    target_table = TARGET_TABLE  # same table you use in /review routes

    # Build OR expression for keywords. Use word boundaries similar to classifier.
    # For BigQuery REGEXP_CONTAINS with \b, use r"\\b" escaping in Python.
    ors = []
    for kw in keywords:
        # word boundary match, case-insensitive: use (?i)
        ors.append(f"REGEXP_CONTAINS(LOWER(Description), r'\\b{re.escape(kw)}\\b')")
    where_kw = " OR ".join(ors)

    # Update rows where currently 'Other' and keywords match
    upd_sql = f"""
    UPDATE `{bq_project}.{bq_dataset}.{target_table}`
    SET Category = @category, MainCategory = @main
    WHERE LOWER(IFNULL(Category, 'other')) = 'other'
      AND ({where_kw})
    """

    client = bq_client()
    job = client.query(
        upd_sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("category", "STRING", category),
                bigquery.ScalarQueryParameter("main", "STRING", main),
            ]
        )
    )
    job.result()

    return redirect(url_for("categories_get", desc=desc, msg=f"Added '{category}' to '{main}' and updated matching rows.", msg_type="ok"))


# ---- Categories tester and updater ----
from pathlib import Path
import json
import re
from typing import List, Dict, Any, Tuple
from flask import request, render_template, redirect, url_for

CATEGORIES_JSON_PATH = Path("categories_grouped.json")

def read_categories() -> List[Dict[str, Any]]:
    with CATEGORIES_JSON_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_categories(cfg: List[Dict[str, Any]]):
    with CATEGORIES_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

def add_category(cfg: List[Dict[str, Any]], main: str, category: str, keywords: List[str]) -> List[Dict[str, Any]]:
    for block in cfg:
        if block.get("main") == main:
            block.setdefault("categories", [])
            for entry in block["categories"]:
                if entry.get("category") == category:
                    existing = {k.strip().lower() for k in entry.get("keywords", []) if k.strip()}
                    for k in keywords:
                        ek = k.strip().lower()
                        if ek:
                            existing.add(ek)
                    entry["keywords"] = sorted(existing)
                    return cfg
            block["categories"].append({"category": category, "keywords": [k.strip().lower() for k in keywords if k.strip()]})
            return cfg
    cfg.append({
        "main": main,
        "categories": [
            {"category": category, "keywords": [k.strip().lower() for k in keywords if k.strip()]}
        ]
    })
    return cfg

def build_regex_index_for_runtime(cfg: List[Dict[str, Any]], use_word_boundaries: bool = True) -> List[Tuple[re.Pattern, str, str]]:
    idx: List[Tuple[re.Pattern, str, str]] = []
    for block in cfg:
        main = block.get("main")
        for entry in block.get("categories", []):
            cat = entry.get("category")
            for kw in entry.get("keywords", []):
                base = kw
                if use_word_boundaries:
                    pat = re.compile(rf"\b{re.escape(base)}\b", re.IGNORECASE)
                else:
                    pat = re.compile(re.escape(base), re.IGNORECASE)
                idx.append((pat, cat, main))
    return idx

def classify_with_index(text: str, regex_index: List[Tuple[re.Pattern, str, str]]) -> Tuple[str, str]:
    t = (text or "").strip()
    for pattern, cat, main in regex_index:
        if pattern.search(t):
            return cat, main
    return "Other", "Other"

@app.get("/categories")
def categories_get():
    desc = request.args.get("desc", "").strip()
    msg = request.args.get("msg", "").strip()
    msg_type = request.args.get("msg_type", "").strip()
    cfg = read_categories()
    regex_index = build_regex_index_for_runtime(cfg)
    cat, main = classify_with_index(desc, regex_index) if desc else ("", "")
    mains = [block.get("main") for block in cfg]
    return render_template(
        "categories.html",
        description=desc,
        result_cat=cat,
        result_main=main,
        mains=mains,
        message=msg,
        message_type=msg_type
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

    cfg = read_categories()
    cfg = add_category(cfg, main, category, keywords)
    write_categories(cfg)

    # Reclassify existing 'Other' rows in BigQuery that match new keywords
    table_id = f"{BQ_PROJECT}.{BQ_DATASET}.{TARGET_TABLE}"

    ors = []
    for kw in keywords:
        # Use LOWER for robustness; boundaries approximate classifier behavior
        ors.append(f"REGEXP_CONTAINS(LOWER(Description), r'\\b{re.escape(kw)}\\b')")
    where_kw = " OR ".join(ors) if ors else "FALSE"

    upd_sql = f"""
    UPDATE `{table_id}`
    SET Category = @category, MainCategory = @main
    WHERE LOWER(IFNULL(Category, 'other')) = 'other'
      AND ({where_kw})
    """

    client = bq_client()
    job = client.query(
        upd_sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("category", "STRING", category),
                bigquery.ScalarQueryParameter("main", "STRING", main),
            ]
        )
    )
    job.result()

    return redirect(url_for("categories_get", desc=desc, msg=f"Added '{category}' under '{main}' and updated matching rows.", msg_type="ok"))



# ---------------- Entrypoint ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
