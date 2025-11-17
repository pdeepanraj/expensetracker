import io
import csv
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

# Existing imports and constants
from pipeline import make_classifier_grouped, run_pipeline
from bq_client import ensure_tables
from bq_write import write_positive_rows
from bq_queries import (
    query_total_by_month,
    query_summary_by_category,
    query_summary_by_main,
    query_latest_year_main_totals,
)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

GITHUB_OWNER = "your-username"
GITHUB_REPO = "your-repo"
GITHUB_BRANCH = "main"

LAST_RESULTS = {}
LAST_RESULTS_JSON = {}

def to_native(obj):
    # Simple helper to ensure JSON-serializable types
    return obj

def read_csv_bytes(content: bytes, filename: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(content))

def gh_headers():
    return {"Accept": "application/vnd.github.v3.raw"}

@app.on_event("startup")
def startup_event():
    # Ensure BigQuery dataset/table exist
    ensure_tables()

@app.get("/", response_class=JSONResponse)
def root():
    return {"status": "ok"}

@app.get("/results", response_class=JSONResponse)
def results_page(request: Request):
    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "GITHUB_OWNER": GITHUB_OWNER,
            "GITHUB_REPO": GITHUB_REPO,
            "GITHUB_BRANCH": GITHUB_BRANCH,
        },
    )

@app.get("/categorized", response_class=JSONResponse)
def categorized_page(request: Request):
    return templates.TemplateResponse(
        "categorized.html",
        {
            "request": request,
            "GITHUB_OWNER": GITHUB_OWNER,
            "GITHUB_REPO": GITHUB_REPO,
            "GITHUB_BRANCH": GITHUB_BRANCH,
        },
    )

@app.post("/categorized-github", response_class=JSONResponse)
async def categorized_github(
    request: Request,
    owner: str = Form(...),
    repo: str = Form(...),
    branch: str = Form("main"),
    urls: str = Form(""),
):
    selected = [u.strip() for u in urls.replace(",", "\n").split("\n") if u.strip()]
    if not selected:
        return templates.TemplateResponse(
            "categorized.html",
            {
                "request": request,
                "error": "please add at least one GitHub raw CSV URL.",
                "GITHUB_OWNER": GITHUB_OWNER,
                "GITHUB_REPO": GITHUB_REPO,
                "GITHUB_BRANCH": GITHUB_BRANCH,
            },
        )

    import requests

    frames: List[pd.DataFrame] = []
    bad: List[str] = []
    for url in selected:
        try:
            r = requests.get(url, timeout=30, headers=gh_headers())
            r.raise_for_status()
            df = read_csv_bytes(r.content, url)
            frames.append(df)
        except Exception as e:
            bad.append(f"{url}: {e}")

    if not frames:
        msg = "No valid CSV loaded from GitHub."
        if bad:
            msg += " Problem files: " + ", ".join(bad)
        return templates.TemplateResponse(
            "categorized.html",
            {
                "request": request,
                "error": msg,
                "GITHUB_OWNER": GITHUB_OWNER,
                "GITHUB_REPO": GITHUB_REPO,
                "GITHUB_BRANCH": GITHUB_BRANCH,
            },
        )

    # Categorize exactly like your existing flow
    classifier = make_classifier_grouped("categories_grouped.json", use_word_boundaries=False)
    result = run_pipeline(frames, classifier)

    # Save in-memory copies as before
    LAST_RESULTS["all_positive_monthly"] = result.get("all_positive_monthly", [])
    LAST_RESULTS["latest_month"] = str(result.get("latest_month", ""))
    LAST_RESULTS["latest_year"] = result.get("latest_year", None)
    LAST_RESULTS["latest_year_main_totals"] = result.get("latest_year_main_totals", [])

    # Write positive rows to BigQuery
    # Build the DataFrame from all_positive_monthly-like structure if needed.
    # However run_pipeline already has clean_data inside; we emulate positive_rows via returned dict
    # For simplicity, recompose a DataFrame from all_positive_monthly records
    apm = result.get("all_positive_monthly", [])
    df = pd.DataFrame(apm or [])
    if not df.empty:
        # Normalize Date and Month fields back if they exist only as strings
        if "Date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Date"]):
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        if "Month" not in df.columns and "Date" in df.columns:
            df["Month"] = df["Date"].dt.to_period("M").astype(str)
        write_positive_rows(df)

    # Instead of computing charts with pandas, read them from BigQuery for performance
    total_by_month = query_total_by_month()
    summary_by_cat = query_summary_by_category()
    summary_by_main = query_summary_by_main()
    latest_year_totals = query_latest_year_main_totals()

    results_json = {
        "latest_month": str(LAST_RESULTS["latest_month"]),
        "total_positive_by_month": total_by_month,
        "summary_by_category": summary_by_cat,
        "summary_by_main": summary_by_main,
        "latest_year": LAST_RESULTS.get("latest_year"),
        "latest_year_main_totals": latest_year_totals,
    }

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "results_json": results_json,
            "GITHUB_OWNER": GITHUB_OWNER,
            "GITHUB_REPO": GITHUB_REPO,
            "GITHUB_BRANCH": GITHUB_BRANCH,
            "warning": f"Skipped {len(bad)} file(s): " + ", ".join(bad) if bad else "",
        },
    )

@app.post("/categorized-analyze", response_class=JSONResponse)
async def categorized_analyze(request: Request, file: UploadFile = File(...)):
    content = await file.read()
    try:
        df = read_csv_bytes(content, file.filename)
    except Exception as e:
        return templates.TemplateResponse(
            "categorized.html", {"request": request, "error": f"Failed to read CSV: {e}"}
        )

    required = {"Month", "CardName", "MainCategory", "Category", "Description", "Amount"}
    if not required.issubset(set(df.columns)):
        return templates.TemplateResponse(
            "categorized.html",
            {
                "request": request,
                "error": f"CSV must include columns: {', '.join(sorted(required))}",
            },
        )

    # Normalize and write rows to BigQuery
    df["Date"] = pd.to_datetime(df.get("Date", None), errors="coerce")
    if df["Date"].isna().all():
        # If Date missing, synthesize from Month as first day
        df["Date"] = pd.to_datetime(df["Month"].astype(str) + "-01", errors="coerce")
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)

    write_positive_rows(df[df["Amount"] > 0])

    info = f"Loaded {len(df)} rows from {file.filename} (positive written to BigQuery)"
    data_json = to_native(df.to_dict(orient="records"))

    return templates.TemplateResponse(
        "categorized.html",
        {
            "request": request,
            "data_json": data_json,
            "info": info,
            "GITHUB_OWNER": GITHUB_OWNER,
            "GITHUB_REPO": GITHUB_REPO,
            "GITHUB_BRANCH": GITHUB_BRANCH,
        },
    )

# Download CSV of last analysis (still supported)
@app.get("/download-positive-expenses")
def download_positive_expenses():
    rows = LAST_RESULTS.get("all_positive_monthly", [])
    if not rows:
        return JSONResponse({"error": "No analysis result available. Run the analyzer first."}, status_code=400)

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "Month", "CardName", "MainCategory", "Category", "Description", "Amount", "Comment"
    ])
    writer.writeheader()
    for r in rows:
        writer.writerow({
            "Month": r.get("Month", ""),
            "CardName": r.get("CardName", ""),
            "MainCategory": r.get("MainCategory", ""),
            "Category": r.get("Category", ""),
            "Description": r.get("Description", ""),
            "Amount": r.get("Amount", ""),
            "Comment": r.get("Comment", ""),
        })
    output.seek(0)
    fname = f"positive_expenses_{LAST_RESULTS.get('latest_month', '')}.csv"
    return StreamingResponse(output, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=\"{fname}\""})

# New API endpoints for charts (consumed by your front-end)
@app.get("/api/total-by-month")
def api_total_by_month():
    return JSONResponse(query_total_by_month())

@app.get("/api/summary-by-category")
def api_summary_by_category(month: Optional[str] = None):
    return JSONResponse(query_summary_by_category(month))

@app.get("/api/summary-by-main")
def api_summary_by_main(month: Optional[str] = None):
    return JSONResponse(query_summary_by_main(month))

@app.get("/api/latest-year-main-totals")
def api_latest_year_main_totals():
    return JSONResponse(query_latest_year_main_totals())
