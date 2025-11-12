# gcs_ingest.py
import os, io, json, datetime as dt
from typing import List, Dict
import pandas as pd
from google.cloud import storage

BUCKET = os.environ["GCS_BUCKET"]                # expense-artifacts-<id>
IN_PREFIX = os.environ.get("GCS_IN_PREFIX", "incoming/")
PROC_PREFIX = os.environ.get("GCS_PROC_PREFIX", "processed/")
ART_PREFIX = os.environ.get("GCS_ART_PREFIX", "artifacts/")

def read_csv_bytes(content: bytes) -> pd.DataFrame:
    encs = [None, "utf-8-sig", "utf-8", "latin1", "utf-16"]
    last = None
    for enc in encs:
        try:
            return pd.read_csv(io.BytesIO(content), encoding=enc if enc else None, engine="python", dtype=str)
        except Exception as e:
            last = e
    raise RuntimeError(f"Unable to parse CSV: {last}")

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    cmap = {
        "month": "Month", "cardname": "CardName", "card_name": "CardName", "card name": "CardName",
        "maincategory": "MainCategory", "main_category": "MainCategory", "main category": "MainCategory",
        "category": "Category", "subcat": "Category", "sub category": "Category", "subcategory": "Category",
        "description": "Description", "desc": "Description",
        "amount": "Amount", "amt": "Amount", "value": "Amount", "total": "Amount",
        "comment": "Comment", "notes": "Comment", "note": "Comment", "remarks": "Comment",
    }
    df = df.rename(columns={c: cmap.get(str(c).strip().lower(), c) for c in df.columns})
    required = ["Month","CardName","MainCategory","Category","Description","Amount"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["Month"] = df["Month"].astype(str).str.strip()
    df["CardName"] = df["CardName"].astype(str).str.strip()
    df["MainCategory"] = df["MainCategory"].astype(str).str.strip()
    df["Category"] = df["Category"].astype(str).str.strip()
    df["Description"] = df["Description"].astype(str).str.strip()
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0).round(2)
    if "Comment" not in df.columns:
        df["Comment"] = ""
    return df[["Month","CardName","MainCategory","Category","Description","Amount","Comment"]]

def summarize(rows: List[Dict]) -> Dict:
    df = pd.DataFrame(rows)
    if df.empty:
        return {
            "latest_month": None,
            "total_rows": 0,
            "totals_by_month": {},
            "totals_by_main": {},
            "totals_by_category": {},
        }
    # latest month by lexical order works with YYYY-MM formats
    latest_month = sorted(df["Month"].astype(str).unique())[-1]
    # totals
    by_month = df.groupby("Month")["Amount"].sum().round(2).to_dict()
    by_main = df.groupby("MainCategory")["Amount"].sum().round(2).to_dict()
    by_cat = df.groupby("Category")["Amount"].sum().round(2).to_dict()
    return {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "latest_month": latest_month,
        "total_rows": int(len(df)),
        "totals_by_month": {k: float(v) for k,v in by_month.items()},
        "totals_by_main": {k: float(v) for k,v in by_main.items()},
        "totals_by_category": {k: float(v) for k,v in by_cat.items()},
    }

def main():
    client = storage.Client()
    bucket = client.bucket(BUCKET)
    blobs = list(client.list_blobs(BUCKET, prefix=IN_PREFIX))
    csvs = [b for b in blobs if b.name.lower().endswith(".csv")]
    if not csvs:
        print("No CSVs to process.")
        return

    all_rows: List[Dict] = []
    for b in csvs:
        print(f"Reading gs://{BUCKET}/{b.name}")
        content = b.download_as_bytes()
        raw = read_csv_bytes(content)
        df = normalize(raw)
        rows = df.to_dict(orient="records")
        # annotate source for traceability
        for r in rows:
            r["_source"] = f"gs://{BUCKET}/{b.name}"
        all_rows.extend(rows)

    # Write artifacts
    rows_path = ART_PREFIX.rstrip("/") + "/rows.jsonl"
    summaries_path = ART_PREFIX.rstrip("/") + "/summaries.json"
    print(f"Writing artifacts to gs://{BUCKET}/{rows_path} and .../summaries.json")

    # rows.jsonl
    rows_blob = bucket.blob(rows_path)
    rows_buf = io.StringIO()
    for r in all_rows:
        rows_buf.write(json.dumps(r, ensure_ascii=False) + "\n")
    rows_blob.upload_from_string(rows_buf.getvalue(), content_type="application/json")

    # summaries.json
    summary = summarize(all_rows)
    sum_blob = bucket.blob(summaries_path)
    sum_blob.upload_from_string(json.dumps(summary, ensure_ascii=False, indent=2), content_type="application/json")

    # Move processed files
    for b in csvs:
        new_name = b.name.replace(IN_PREFIX, PROC_PREFIX, 1) if b.name.startswith(IN_PREFIX) else f"{PROC_PREFIX}{b.name}"
        bucket.rename_blob(b, new_name)
        print(f"Moved {b.name} -> {new_name}")

if __name__ == "__main__":
    main()
