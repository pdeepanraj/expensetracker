from google.cloud import bigquery

DATASET_ID = "expense_analytics"
TABLE_ID = "transactions"

def get_bq_client() -> bigquery.Client:
    # Uses application default credentials in Cloud Run/App Engine
    return bigquery.Client()

def ensure_tables():
    client = get_bq_client()
    project = client.project

    # Create dataset if it doesn't exist
    ds_ref = bigquery.Dataset(f"{project}.{DATASET_ID}")
    client.create_dataset(ds_ref, exists_ok=True)

    # Define table schema
    schema = [
        bigquery.SchemaField("Date", "TIMESTAMP"),
        bigquery.SchemaField("Month", "STRING"),
        bigquery.SchemaField("CardName", "STRING"),
        bigquery.SchemaField("MainCategory", "STRING"),
        bigquery.SchemaField("Category", "STRING"),
        bigquery.SchemaField("Description", "STRING"),
        bigquery.SchemaField("Amount", "FLOAT"),
        bigquery.SchemaField("Comment", "STRING"),
        bigquery.SchemaField("RowId", "STRING"),
    ]

    table = bigquery.Table(f"{project}.{DATASET_ID}.{TABLE_ID}", schema=schema)
    table.time_partitioning = bigquery.TimePartitioning(field="Date")
    table.clustering_fields = ["MainCategory", "Category", "CardName"]

    client.create_table(table, exists_ok=True)
