import io
import csv
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

# Import from the app package
from app.pipeline import make_classifier_grouped, run_pipeline
from app.bq_client import ensure_tables
from app.bq_write import write_positive_rows
from app.bq_queries import (
    query_total_by_month,
    query_summary_by_category,
    query_summary_by_main,
    query_latest_year_main_totals,
)
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")

# -----------------------------------------------------------------------------
# FastAPI app and template setup
# -----------------------------------------------------------------------------

app = FastAPI()

# Ensure this points to the templates folder inside app/
templates = Jinja2Templates(directory="app/templates")

# Optional defaults displayed on the UI
GITHUB_OWNER = "your-username"
GITHUB_REPO = "your-repo"
GITHUB_BRANCH = "main"

# Keep a small cache for "download" and UI info, as in your original app
LAST_RESULTS: dict = {}
LAST_RESULTS_JSON: dict = {}

# Helper to resolve paths relative to this file
APP_DIR = Path(__file__).parent
CATEGORIES_JSON_PATH = str(APP_DIR / "categories_grouped.json")


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def to_native(obj):
    # In case you want to do special conversions before sending to templates
    return obj

def read_csv_bytes(content: bytes, filename: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(content))

def gh_headers():
    # If you have a token, you can add Authorization header; raw works for public
    return {"Accept": "application/vnd.github.v3.raw"}


# -----------------------------------------------------------------------------
# Startup: ensure BigQuery dataset/table exist
# -----------------------------------------------------------------------------

@app.on_event("startup")
def startup_event():
    ensure_tables()


# -----------------------------------------------------------------------------
# Basic endpoints
# -----------------------------------------------------------------------------

@app.get("/", response_class=JSONResponse)
def root():
    return {"status": "ok"}

@app.get("/results")
def results_page(request: Request):
    # You can render without data; the front-end can fetch /api/* for charts
    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "GITHUB_OWNER": GITHUB_OWNER,
            "GITHUB_REPO": GITHUB_REPO,
            "GITHUB_BRANCH": GITHUB_BRANCH,
        },
    )

@app.get("/categorized")
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


# -----------------------------------------------------------------------------
# Analyze CSVs fetched from GitHub
# -----------------------------------------------------------------------------

@app.post("/categorized-github")
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

    # Categorize using the original logic
    classifier = make_classifier_grouped(CATEGORIES_JSON_PATH, use_word_boundaries=False)
    result = run_pipeline(frames, classifier)

    # Persist minimal info locally (for download endpoint)
    LAST_RESULTS["all_positive_monthly"] = result.get("all_positive_monthly", [])
    LAST_RESULTS["latest_month"] = str(result.get("latest_month", ""))
    LAST_RESULTS["latest_year"] = result.get("latest_year", None)
    LAST_RESULTS["latest_year_main_totals"] = result.get("latest_year_main_totals", [])

    # Write positive rows to BigQuery
    # Reconstruct a DataFrame from returned records (if not already keeping df)
    apm = result.get("all_positive_monthly", [])
    df_out = pd.DataFrame(apm or [])
    if not df_out.empty:
        # Normalize Date and Month columns
        if "Date" in df_out.columns and not pd.api.types.is_datetime64_any_dtype(df_out["Date"]):
            df_out["Date"] = pd.to_datetime(df_out["Date"], errors="coerce")
        if "Month" not in df_out.columns:
            if "Date" in df_out.columns:
                df_out["Month"] = df_out["Date"].dt.to_period("M").astype(str)
            else:
                # If Month is missing and Date unavailable, skip insert to avoid bad rows
                pass
        # Insert only positive amounts
        if "Amount" in df_out.columns:
            df_out["Amount"] = pd.to_numeric(df_out["Amount"], errors="coerce").fillna(0.0)
            df_pos = df_out[df_out["Amount"] > 0].copy()
            if not df_pos.empty:
                write_positive_rows(df_pos)

    # Fetch aggregates from BigQuery
    total_by_month = query_total_by_month()
    summary_by_cat = query_summary_by_category()
    summary_by_main_res = query_summary_by_main()
    latest_year_totals = query_latest_year_main_totals()

    results_json = {
        "latest_month": str(LAST_RESULTS.get("latest_month", "")),
        "total_positive_by_month": total_by_month,
        "summary_by_category": summary_by_cat,
        "summary_by_main": summary_by_main_res,
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


# -----------------------------------------------------------------------------
# Analyze uploaded categorized CSV
# -----------------------------------------------------------------------------

@app.post("/categorized-analyze")
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

    # Normalize core fields
    # Date may be absent in the uploaded CSV; synthesize from Month if needed
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    else:
        df["Date"] = pd.to_datetime(df["Month"].astype(str) + "-01", errors="coerce")

    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)

    # Write positive rows to BigQuery
    df_pos = df[df["Amount"] > 0].copy()
    if not df_pos.empty:
        write_positive_rows(df_pos)

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


# -----------------------------------------------------------------------------
# Download CSV of last in-memory analysis (kept for parity with original app)
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# API endpoints for charts backed by BigQuery
# -----------------------------------------------------------------------------

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
