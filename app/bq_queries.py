from google.cloud import bigquery
from typing import Optional, List, Dict

DATASET_ID = "expense_analytics"
TABLE_ID = "transactions"

def _table(project: str) -> str:
    return f"`{project}.{DATASET_ID}.{TABLE_ID}`"

def query_total_by_month() -> List[Dict]:
    client = bigquery.Client()
    sql = f"""
    SELECT FORMAT_TIMESTAMP('%Y-%m', Date) AS Month, SUM(Amount) AS Amount
    FROM {_table(client.project)}
    WHERE Amount > 0
    GROUP BY Month
    ORDER BY Month;
    """
    return [dict(row) for row in client.query(sql).result()]

def query_summary_by_category(month: Optional[str] = None) -> List[Dict]:
    client = bigquery.Client()
    if month:
        sql = f"""
        SELECT FORMAT_TIMESTAMP('%Y-%m', Date) AS Month, Category, SUM(Amount) AS Amount
        FROM {_table(client.project)}
        WHERE Amount > 0 AND FORMAT_TIMESTAMP('%Y-%m', Date) = @month
        GROUP BY Month, Category
        ORDER BY Amount DESC;
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("month", "STRING", month)]
        )
        result = client.query(sql, job_config=job_config).result()
    else:
        sql = f"""
        SELECT FORMAT_TIMESTAMP('%Y-%m', Date) AS Month, Category, SUM(Amount) AS Amount
        FROM {_table(client.project)}
        WHERE Amount > 0
        GROUP BY Month, Category
        ORDER BY Month DESC, Amount DESC;
        """
        result = client.query(sql).result()
    return [dict(row) for row in result]

def query_summary_by_main(month: Optional[str] = None) -> List[Dict]:
    client = bigquery.Client()
    if month:
        sql = f"""
        SELECT FORMAT_TIMESTAMP('%Y-%m', Date) AS Month, MainCategory, SUM(Amount) AS Amount
        FROM {_table(client.project)}
        WHERE Amount > 0 AND FORMAT_TIMESTAMP('%Y-%m', Date) = @month
        GROUP BY Month, MainCategory
        ORDER BY Amount DESC;
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("month", "STRING", month)]
        )
        result = client.query(sql, job_config=job_config).result()
    else:
        sql = f"""
        SELECT FORMAT_TIMESTAMP('%Y-%m', Date) AS Month, MainCategory, SUM(Amount) AS Amount
        FROM {_table(client.project)}
        WHERE Amount > 0
        GROUP BY Month, MainCategory
        ORDER BY Month DESC, Amount DESC;
        """
        result = client.query(sql).result()
    return [dict(row) for row in result]

def query_latest_year_main_totals() -> List[Dict]:
    client = bigquery.Client()
    sql = f"""
    WITH latest_y AS (
      SELECT EXTRACT(YEAR FROM MAX(Date)) AS y
      FROM {_table(client.project)}
      WHERE Amount > 0
    )
    SELECT t.MainCategory, SUM(t.Amount) AS Amount
    FROM {_table(client.project)} t, latest_y ly
    WHERE t.Amount > 0 AND EXTRACT(YEAR FROM t.Date) = ly.y
    GROUP BY t.MainCategory
    ORDER BY Amount DESC;
    """
    return [dict(row) for row in client.query(sql).result()]
