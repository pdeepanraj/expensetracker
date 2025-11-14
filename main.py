import os
import io
import csv
import re
import json
import httpx
from datetime import datetime
from typing import List, Dict, Any, Tuple

import pandas as pd
from fastapi import FastAPI, Query, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from google.cloud import bigquery

# ------------ Config ------------
BQ_PROJECT = os.environ.get("BQ_PROJECT")            # e.g., "my-gcp-project"
BQ_DATASET = os.environ.get("BQ_DATASET", "expense_ds")
BQ_TABLE   = os.environ.get("BQ_TABLE", "transactions")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")        # optional

# ------------ App init ------------
app = FastAPI(title="Expense Tracker")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ------------ Categorizer ------------
EXCLUDE_KEYWORDS = ['citi flex','online payment','billprotect','ach dep','ach bt']

def load_grouped_category_config(text: str) -> List[Dict[str, Any]]:
    config = json.loads(text)
    # light validation
    for block in config:
        if "main" not in block or "categories" not in block:
            raise ValueError("Invalid category block")
        for cat in block["categories"]:
            if "category" not in cat or "keywords" not in cat:
                raise ValueError("Invalid category entry")
    return config

def build_regex_index(config) -> List[Tuple[re.Pattern, str, str]]:
    idx = []
    for block in config:
        m = block["main"]
        for c in block["categories"]:
            sub = c["category"]
            for kw in c["keywords"]:
                pat = re.compile(rf"\b{re.escape(kw.lower())}\b")
                idx.append((pat, sub, m))
    return idx

def make_classifier(config_text: str):
    config = load_grouped_category_config(config_text)
    idx = build_regex_index(config)
    def classify(desc: str) -> Tuple[str, str]:
        d = (desc or "").lower()
        for pat, sub, main in idx:
            if pat.search(d):
                return sub, main
        return "Other", "Other"
    return classify

# ------------ CSV ingestion ------------
def _headers():
    h = {"Accept": "text/plain"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h

async def fetch_csv(url: str) -> pd.DataFrame:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, headers=_headers())
        r.raise_for_status()
        return pd.read_csv(io.BytesIO(r.content))

def normalize_amount_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [str(c).strip() for c in df.columns]
    df.columns = cols
    lc = {c.lower(): c for c in cols}
    def numify(s):
        return pd.to_numeric(s, errors="coerce").fillna(0)

    if 'amount' in lc:
        df['Amount'] = numify(df[lc['amount']])
    elif 'debit' in lc:
        df['Amount'] = numify(df[lc['debit']])
    elif 'credit' in lc:
        df['Amount'] = -numify(df[lc['credit']])  # credits as negative
    else:
        raise ValueError("No recognizable amount column (Amount/Debit/Credit)")

    return df

def detect_and_standardize(df: pd.DataFrame, card_name: str, exclude=EXCLUDE_KEYWORDS) -> pd.DataFrame:
    df = normalize_amount_columns(df)
    # detect date/description
    date_col = next((c for c in df.columns if str(c).lower() == 'date'), None)
    if not date_col:
        date_col = next((c for c in df.columns if 'date' in str(c).lower()), None)
    desc_col = next((c for c in df.columns if str(c).lower() in ('description','desc')), None)
    if not date_col or not desc_col:
        raise ValueError("Expected Date and Description columns")

    out = df.rename(columns={date_col:'Date', desc_col:'Description'})[['Date','Description','Amount']].copy()
    # filter noise
    dl = out['Description'].astype(str).str.lower()
    for kw in exclude:
        out = out[~dl.str.contains(kw, na=False)]
        dl = out['Description'].astype(str).str.lower()

    out['Date'] = pd.to_datetime(out['Date'], errors='coerce')
    out = out.dropna(subset=['Date'])
    out['Month'] = out['Date'].dt.to_period('M').astype(str)
    out['CardName'] = card_name
    return out

def remove_offsetting_pairs(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp['AbsAmount'] = tmp['Amount'].abs()
    grp = (tmp.groupby(['Month','CardName','Description'])
             .agg(minA=('Amount','min'), maxA=('Amount','max'))
             .reset_index())
    pair_keys = grp[(grp['minA'] < 0) & (grp['maxA'] > 0)][['Month','CardName','Description']]

    keyset = set(map(tuple, pair_keys.to_records(index=False)))
    mask = tmp.apply(lambda r: (r['Month'], r['CardName'], r['Description']) in keyset, axis=1)
    return tmp[~mask].drop(columns=['AbsAmount'], errors='ignore')

# ------------ BigQuery I/O ------------
bq_client = bigquery.Client()  # uses default credentials

def to_bq_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return [
        {
            "txn_id": f"{r.Date:%Y%m%d}-{hash((r.CardName, r.Description, float(r.Amount), r.Date)) & 0xffffffff}",
            "date": r.Date.date().isoformat(),
            "month": r.Month,
            "card_name": r.CardName,
            "description": r.Description,
            "main_category": r.MainCategory,
            "sub_category": r.Category,
            "amount": float(r.Amount),
            "comment": r.get("Comment",""),
            "source_file": r.get("SourceFile",""),
        }
        for _, r in df.iterrows()
    ]

def load_to_bq(df: pd.DataFrame):
    table_ref = f"{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"
    errors = bq_client.insert_rows_json(table_ref, to_bq_rows(df))
    if errors:
        raise HTTPException(status_code=500, detail=f"BigQuery insert errors: {errors}")

# ------------ Routes ------------
@app.get("/", response_class=HTMLResponse)
def home():
    with open("static/index.html","r",encoding="utf-8") as f:
        return f.read()

@app.post("/ingest-from-github")
async def ingest_from_github(
    csv_urls: List[str] = Query(..., description="One or more raw GitHub CSV URLs"),
    card_name: str = Query("Uploaded"),
    categories_json_url: str = Query(..., description="Raw GitHub URL to categories_grouped.json")
):
    # fetch categories
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(categories_json_url, headers=_headers())
        resp.raise_for_status()
        cat_text = resp.text
    classify = make_classifier(cat_text)

    frames = []
    bad = []
    for u in csv_urls:
        try:
            df = await fetch_csv(u)
            df = detect_and_standardize(df, card_name)
            df['SourceFile'] = u
            frames.append(df)
        except Exception as e:
            bad.append(f"{u}: {e}")

    if not frames:
        raise HTTPException(status_code=400, detail=f"No CSVs ingested. Errors: {bad}")

    data = pd.concat(frames, ignore_index=True)
    data = remove_offsetting_pairs(data)
    # Categorize
    cats = data['Description'].apply(lambda d: classify(str(d)))
    data['Category'] = cats.apply(lambda x: x[0])
    data['MainCategory'] = cats.apply(lambda x: x[1])
    data['Comment'] = data.apply(lambda r: r['Description'] if r['Category']=="Other" else "", axis=1)

    # Load to BQ
    load_to_bq(data)

    return {"rows_loaded": int(len(data)), "errors": bad}

@app.get("/api/summary")
def api_summary(
    group: str = Query("month,main_category"),
    agg: str = Query("sum")  # sum, count, mean, min, max, median
):
    # Build SQL safely from allowed fields
    allowed = {
        "month":"month", "year":"EXTRACT(YEAR FROM date) AS year",
        "main_category":"main_category", "sub_category":"sub_category",
        "card_name":"card_name"
    }
    group_keys = [g.strip() for g in group.split(",") if g.strip()]
    if not group_keys:
        raise HTTPException(status_code=400, detail="Provide at least one group key")

    select_parts = []
    group_cols = []
    for g in group_keys:
        if g not in allowed:
            raise HTTPException(status_code=400, detail=f"Invalid group key: {g}")
        if g == "year":
            select_parts.append(allowed[g])
            group_cols.append("year")
        else:
            select_parts.append(allowed[g])
            group_cols.append(g)

    agg_expr = {
        "sum": "SUM(amount) AS value",
        "count": "COUNT(1) AS value",
        "mean": "AVG(amount) AS value",
        "min": "MIN(amount) AS value",
        "max": "MAX(amount) AS value",
        "median": "APPROX_QUANTILES(amount, 2)[OFFSET(1)] AS value"
    }.get(agg)
    if not agg_expr:
        raise HTTPException(status_code=400, detail="Invalid agg")

    sql = f"""
    SELECT {', '.join(select_parts)}, {agg_expr}
    FROM `{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}`
    GROUP BY {', '.join(group_cols)}
    ORDER BY {', '.join(group_cols)}
    """
    df = bq_client.query(sql).to_dataframe()
    return {"group": group_cols, "agg": agg, "rows": df.to_dict(orient="records")}
