from google.cloud import bigquery
import hashlib
from typing import Iterable
import pandas as pd

DATASET_ID = "expense_analytics"
TABLE_ID = "transactions"

def _make_row_id(date_val, card, desc, amount) -> str:
    base = f"{date_val}|{card}|{desc}|{amount}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()

def write_positive_rows(df: pd.DataFrame):
    """Stream categorized positive transactions to BigQuery."""
    client = bigquery.Client()
    table_id = f"{client.project}.{DATASET_ID}.{TABLE_ID}"

    rows: Iterable[dict] = []
    for _, r in df.iterrows():
        dt = r["Date"]
        # Convert to native datetime for BigQuery
        dt_py = dt.to_pydatetime() if hasattr(dt, "to_pydatetime") else dt
        month_str = str(r["Month"])
        rows = list(rows)  # in case generator; ensure list accumulation
        rows.append({
            "Date": dt_py,
            "Month": month_str,
            "CardName": r.get("CardName", ""),
            "MainCategory": r.get("MainCategory", ""),
            "Category": r.get("Category", ""),
            "Description": r.get("Description", ""),
            "Amount": float(r.get("Amount", 0.0)),
            "Comment": r.get("Comment", ""),
            "RowId": _make_row_id(month_str, r.get("CardName", ""), r.get("Description", ""), r.get("Amount", 0.0)),
        })

    # Insert rows; optional: de-dup by RowId via MERGE pattern
    errors = client.insert_rows_json(table_id, rows)
    if errors:
        # You could choose to log and continue; here we fail to surface issues
        raise RuntimeError(f"BigQuery insert errors: {errors}")
