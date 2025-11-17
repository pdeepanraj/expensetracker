from fastapi import FastAPI, Request, Query, Form, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import io
import os
import requests
import csv
from typing import List, Optional

from pipeline import run_pipeline, make_classifier_grouped, clean_and_standardize

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# GitHub defaults via env (optional)
GITHUB_OWNER = os.environ.get("GITHUB_OWNER", "")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "")
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

# Last analysis for download
LAST_RESULTS = {
    "all_positive_monthly": [],
    "latest_month": None
}

def gh_headers():
    h = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h

def read_csv_bytes(content: bytes, filename: str) -> pd.DataFrame:
    if not content:
        raise pd.errors.EmptyDataError("empty file")
    encodings = [None, "utf-8", "utf-16", "latin1"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            if enc:
                return pd.read_csv(io.BytesIO(content), encoding=enc)
            else:
                return pd.read_csv(io.BytesIO(content))
        except pd.errors.EmptyDataError as e:
            raise e
        except Exception as e:
            last_err = e
            continue
    raise ValueError(f"Unable to parse {filename} with common encodings: {last_err}")

def list_github_csvs(owner: str, repo: str, path: str, branch: str = "main") -> List[str]:
    # Allow empty path (root) and normalize without leading slash for GitHub API
    path = path.strip("/")
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}" if path else f"https://api.github.com/repos/{owner}/{repo}/contents"
    resp = requests.get(url, params={"ref": branch}, headers=gh_headers(), timeout=20)
    if resp.status_code != 200:
        raise RuntimeError(f"GitHub list error {resp.status_code}: {resp.text}")
    data = resp.json()
    items: List[str] = []
    if isinstance(data, list):
        for obj in data:
            if obj.get("type") == "file" and obj.get("name", "").lower().endswith(".csv"):
                if obj.get("download_url"):
                    items.append(obj["download_url"])
    else:
        if data.get("type") == "file" and data.get("name", "").lower().endswith(".csv") and data.get("download_url"):
            items.append(data["download_url"])
    return items

def read_github_csv(raw_url: str) -> pd.DataFrame:
    r = requests.get(raw_url, timeout=30, headers=gh_headers())
    r.raise_for_status()
    content = r.content
    return read_csv_bytes(content, raw_url)

def to_native(obj):
    import numpy as np
    import pandas as pd
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    try:
        import pandas as pd
        if isinstance(obj, pd.Period):
            return str(obj)
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
    except Exception:
        pass
    return str(obj)

# ------------- Routes -------------

# Mode selection
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# GitHub listing and analysis page
@app.get("/github", response_class=HTMLResponse)
def github_page(request: Request):
    return templates.TemplateResponse(
        "github.html",
        {
            "request": request,
            "GITHUB_OWNER": GITHUB_OWNER,
            "GITHUB_REPO": GITHUB_REPO,
            "GITHUB_BRANCH": GITHUB_BRANCH,
        },
    )

@app.get("/gh-list")
def gh_list(
    owner: str = Query(...),
    repo: str = Query(...),
    path: str = Query(""),
    branch: str = Query("main"),
):
    try:
        print(f"[gh-list] owner={owner} repo={repo} path={path} branch={branch}")
        items = list_github_csvs(owner, repo, path, branch)
        print(f"[gh-list] found {len(items)} CSV(s)")
        return JSONResponse({"items": items})
    except Exception as e:
        print(f"[gh-list] error: {e}")
        return JSONResponse({"error": str(e)}, status_code=400)

@app.post("/analyze-github", response_class=HTMLResponse)
async def analyze_github(
    request: Request,
    owner: str = Form(...),
    repo: str = Form(...),
    branch: str = Form("main"),
    urls: str = Form(""),
):
    selected = [u.strip() for u in urls.replace(",", "\n").split("\n") if u.strip()]
    if not selected:
        return templates.TemplateResponse(
            "github.html",
            {
                "request": request,
                "error": "Please add at least one GitHub raw CSV URL.",
                "GITHUB_OWNER": GITHUB_OWNER,
                "GITHUB_REPO": GITHUB_REPO,
                "GITHUB_BRANCH": GITHUB_BRANCH,
            },
        )

    std_frames: List[pd.DataFrame] = []
    bad_files: List[str] = []
    for url in selected:
        try:
            df = read_github_csv(url)
            df.columns = [str(c).strip().lstrip('\ufeff') for c in df.columns]
            from urllib.parse import urlparse
            from pathlib import Path

            # ...
            parsed = urlparse(url)
            # Path from URL path component; strip query already handled by urlparse
            fname = Path(parsed.path).name  # e.g., 'Amex_BC_OCT25.csv'
            base = Path(fname).stem         # e.g., 'Amex_BC_OCT25'
            card_name = base or "Unknown"
            
            std_frames.append(clean_and_standardize(df, card_name=card_name))
        except Exception as e:
            bad_files.append(f"{url}: {e}")

    if not std_frames:
        msg = "No valid CSV loaded from GitHub."
        if bad_files:
            msg += " Problem files: " + ", ".join(bad_files)
        return templates.TemplateResponse(
            "github.html",
            {
                "request": request,
                "error": msg,
                "GITHUB_OWNER": GITHUB_OWNER,
                "GITHUB_REPO": GITHUB_REPO,
                "GITHUB_BRANCH": GITHUB_BRANCH,
            },
        )

    classifier = make_classifier_grouped("categories_grouped.json", use_word_boundaries=False)
    result = run_pipeline(std_frames, classifier)
    warning = f"Skipped {len(bad_files)} file(s): {', '.join(bad_files)}" if bad_files else None

    LAST_RESULTS["all_positive_monthly"] = result.get("all_positive_monthly", [])
    LAST_RESULTS["latest_month"] = str(result.get("latest_month", "")) if result.get("latest_month") is not None else ""

    results_json = {
        "latest_month": str(result["latest_month"]),
        "positive_monthly": result["positive_monthly"],
        "summary_by_category": result["summary_by_category"],
        "summary_by_main": result["summary_by_main"],
        "total_positive_by_month": result["total_positive_by_month"],
        "latest_year": result["latest_year"],
        "latest_year_main_totals": result["latest_year_main_totals"],
        "all_positive_monthly": result["all_positive_monthly"],
    }
    results_json = to_native(results_json)

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "latest_month": str(result["latest_month"]),
            "positive_monthly": result["positive_monthly"],
            "summary_by_category": result["summary_by_category"],
            "summary_by_main": result["summary_by_main"],
            "total_positive_by_month": result["total_positive_by_month"],
            "latest_year": result["latest_year"],
            "latest_year_main_totals": result["latest_year_main_totals"],
            "uploaded_uris": selected,
            "warning": warning,
            "results_json": results_json,
        },
    )

# Categorized (GitHub) page
@app.get("/categorized", response_class=HTMLResponse)
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

@app.post("/categorized-github", response_class=HTMLResponse)
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
                "error": "Please add at least one GitHub raw CSV URL.",
                "GITHUB_OWNER": GITHUB_OWNER,
                "GITHUB_REPO": GITHUB_REPO,
                "GITHUB_BRANCH": GITHUB_BRANCH,
            },
        )

    # Fetch and combine all selected CSVs
    frames = []
    bad = []
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

    df = pd.concat(frames, ignore_index=True)

    # Require categorized schema
    required = {'Month', 'CardName', 'MainCategory', 'Category', 'Description', 'Amount'}
    missing = required - set(df.columns)
    if missing:
        return templates.TemplateResponse(
            "categorized.html",
            {
                "request": request,
                "error": f"CSV missing required columns: {', '.join(sorted(missing))}. Expected: {', '.join(sorted(required))}",
                "GITHUB_OWNER": GITHUB_OWNER,
                "GITHUB_REPO": GITHUB_REPO,
                "GITHUB_BRANCH": GITHUB_BRANCH,
            },
        )

    # Normalize types
    df['Month'] = df['Month'].astype(str)
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0).astype(float)
    for col in ['CardName', 'MainCategory', 'Category', 'Description']:
        df[col] = df[col].astype(str)
    if 'Comment' not in df.columns:
        df['Comment'] = ''

    data_json = to_native(df.to_dict(orient='records'))
    info = f"Loaded {len(df)} rows from {len(frames)} file(s)"
    if bad:
        info += f" | Skipped {len(bad)} file(s): {', '.join(bad)}"

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

# Upload (categorized CSV) page
@app.post("/categorized-analyze", response_class=HTMLResponse)
async def categorized_analyze(request: Request, file: UploadFile = File(...)):
    content = await file.read()
    try:
        df = read_csv_bytes(content, file.filename or "categorized.csv")
    except Exception as e:
        return templates.TemplateResponse("categorized.html", {"request": request, "error": f"Failed to read CSV: {e}"})

    required = {'Month', 'CardName', 'MainCategory', 'Category', 'Description', 'Amount'}
    if not required.issubset(set(df.columns)):
        return templates.TemplateResponse("categorized.html", {"request": request, "error": f"CSV must include columns: {', '.join(sorted(required))}"})

    df['Month'] = df['Month'].astype(str)
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)

    data_json = to_native(df.to_dict(orient='records'))
    return templates.TemplateResponse(
        "categorized.html",
        {
            "request": request,
            "data_json": data_json,
            "info": f"Loaded {len(df)} rows from {file.filename}",
            "GITHUB_OWNER": GITHUB_OWNER,
            "GITHUB_REPO": GITHUB_REPO,
            "GITHUB_BRANCH": GITHUB_BRANCH,
        },
    )

# Download CSV (from last GitHub analysis)
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
    filename = f'positive_expenses_{LAST_RESULTS.get("latest_month", "")}.csv'
    return StreamingResponse(output, media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="{filename}"'
    })

# Net Expense page
from fastapi import Form

@app.get("/net-expense", response_class=HTMLResponse)
def net_expense_page(request: Request):
    # Initial render with empty values
    return templates.TemplateResponse(
        "net_expense.html",
        {
            "request": request,
            "salary": "",
            "ccpay": "",
            "other": "",
            "total_expense": None,
            "balance": None,
            "error": None,
        },
    )

@app.post("/net-expense", response_class=HTMLResponse)
async def net_expense_calc(
    request: Request,
    salary: List[str] = Form([]),   # salary[]
    ccpay: List[str] = Form([]),    # ccpay[]
    other: str = Form(""),
):
    def to_amount(s: str) -> float:
        s = (s or "").replace(",", "").replace("$", "").strip()
        if s == "":
            return 0.0
        return float(s)

    error = None
    total_expense = None
    balance = None

    try:
        total_salary = round(sum(to_amount(x) for x in salary), 2)
        total_cc = round(sum(to_amount(x) for x in ccpay), 2)
        other_amt = round(to_amount(other), 2)

        total_expense = round(total_cc + other_amt, 2)
        balance = round(total_salary - total_expense, 2)
    except Exception as e:
        error = f"{e}"

    return templates.TemplateResponse(
        "net_expense.html",
        {
            "request": request,
            "salary": salary,
            "ccpay": ccpay,
            "other": other,
            "total_expense": total_expense,
            "balance": balance,
            "error": error,
        },
    )
