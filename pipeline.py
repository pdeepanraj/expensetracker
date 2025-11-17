import pandas as pd
import json
import re
from pathlib import Path
from typing import List, Tuple

EXCLUDE_KEYWORDS = ['citi flex', 'online payment', 'biltprotect rent ach credit', 'ach bt']

def load_grouped_category_config(json_path: str):
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Category JSON not found at: {path.resolve()}")
    with path.open('r', encoding='utf-8') as f:
        config = json.load(f)
    for main_block in config:
        if 'main' not in main_block or 'categories' not in main_block:
            raise ValueError(f"Invalid main block: {main_block}")
        if not isinstance(main_block['categories'], list):
            raise ValueError(f"'categories' must be a list in block: {main_block}")
        for entry in main_block['categories']:
            if not {'category', 'keywords'} <= set(entry.keys()):
                raise ValueError(f"Invalid category entry: {entry}")
            if not isinstance(entry['keywords'], list) or not all(isinstance(k, str) for k in entry['keywords']):
                raise ValueError(f"'keywords' must be a list of strings in: {entry}")
    return config

def build_regex_index_grouped(config, use_word_boundaries: bool = True):
    regex_index: List[Tuple[re.Pattern, str, str]] = []
    for main_block in config:
        main = main_block['main']
        for entry in main_block['categories']:
            cat = entry['category']
            for kw in entry['keywords']:
                base = kw
                if use_word_boundaries and re.fullmatch(r'[\w\s]+', base.lower()):
                    pattern = re.compile(rf'\b{re.escape(base)}\b', re.IGNORECASE)
                else:
                    # Plain substring, case-insensitive
                    pattern = re.compile(re.escape(base), re.IGNORECASE)
                regex_index.append((pattern, cat, main))
    return regex_index

def make_classifier_grouped(config_path: str, use_word_boundaries: bool = True):
    config = load_grouped_category_config(config_path)
    regex_index = build_regex_index_grouped(config, use_word_boundaries=use_word_boundaries)
    def classify(desc: str):
        text = (desc or "")
        for pattern, cat, main in regex_index:
            if pattern.search(text):
                return cat, main
        return "Other", "Other"
    return classify

def normalize_amount_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}
    if 'debit' in lower_map or 'credit' in lower_map:
        debit = df[lower_map['debit']].fillna(0) if 'debit' in lower_map else 0
        credit = df[lower_map['credit']].fillna(0) if 'credit' in lower_map else 0
        df['Amount'] = pd.to_numeric(debit, errors='coerce').fillna(0) + pd.to_numeric(credit, errors='coerce').fillna(0)
    elif 'amount' in lower_map:
        df['Amount'] = pd.to_numeric(df[lower_map['amount']], errors='coerce').fillna(0)
    else:
        raise ValueError("No recognizable amount columns (Debit/Credit/Amount) found.")
    return df

def find_date_desc_columns(df: pd.DataFrame):
    date_col = next((c for c in df.columns if c.lower() == 'date'), None)
    desc_col = next((c for c in df.columns if c.lower() == 'description'), None)
    if date_col is None:
        date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    if desc_col is None:
        desc_col = next((c for c in df.columns if 'desc' in c.lower() or 'description' in c.lower()), None)
    if date_col is None or desc_col is None:
        raise ValueError("Expected Date/Description columns not found.")
    return date_col, desc_col

def clean_and_standardize(df: pd.DataFrame, card_name: str = "Uploaded") -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    df = normalize_amount_columns(df)

    date_col, desc_col = find_date_desc_columns(df)
    df = df.rename(columns={date_col: 'Date', desc_col: 'Description'})
    df = df[['Date', 'Description', 'Amount']]

    desc_lower = df['Description'].astype(str).str.lower()
    for keyword in EXCLUDE_KEYWORDS:
        df = df[~desc_lower.str.contains(keyword, na=False)]
        desc_lower = df['Description'].astype(str).str.lower()

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['Month'] = df['Date'].dt.to_period('M')

    df['CardName'] = card_name
    return df

def run_pipeline(frames: List[pd.DataFrame], classifier):
    # frames are already standardized
    all_data = pd.concat(frames, ignore_index=True)

    # Categorize and add comments
    all_data[['Category', 'MainCategory']] = all_data['Description'].apply(lambda d: pd.Series(classifier(d)))
    all_data['Comment'] = all_data.apply(lambda r: r['Description'] if r['Category'] == 'Other' else '', axis=1)

    # Remove offsetting positive/negative pairs at transaction level
    all_data['AbsAmount'] = all_data['Amount'].abs()
    pair_stats = (all_data.groupby(['Month', 'CardName', 'MainCategory', 'Category', 'AbsAmount'])
                  .agg(MinAmount=('Amount', 'min'), MaxAmount=('Amount', 'max'))
                  .reset_index())
    offset_pairs = pair_stats[(pair_stats['MinAmount'] < 0) & (pair_stats['MaxAmount'] > 0)]
    offset_key = set(tuple(x) for x in offset_pairs[['Month','CardName','MainCategory','Category','AbsAmount']].to_records(index=False))

    mask_exclude = all_data.apply(
        lambda r: (r['Month'], r['CardName'], r['MainCategory'], r['Category'], r['AbsAmount']) in offset_key,
        axis=1
    )
    clean_data = all_data[~mask_exclude].copy()

    positive_rows = clean_data[clean_data['Amount'] > 0].copy()
    latest_month = positive_rows['Month'].max()

    def agg_comment(s):
        vals = [v for v in s if isinstance(v, str) and v.strip()]
        if not vals:
            return ''
        uniq = list(dict.fromkeys(vals))
        return '; '.join(uniq)

    # Latest month details
    latest_positive_monthly = (positive_rows[positive_rows['Month'] == latest_month]
        .groupby(['Month', 'CardName', 'MainCategory', 'Category', 'Description'], as_index=False)
        .agg(Amount=('Amount', 'sum'),
             Comment=('Comment', agg_comment))
    )

    # All months details (for client-side filtering)
    all_positive_monthly = (positive_rows
        .groupby(['Month', 'CardName', 'MainCategory', 'Category', 'Description'], as_index=False)
        .agg(Amount=('Amount', 'sum'),
             Comment=('Comment', agg_comment))
    )

    summary_by_category = (positive_rows
        .groupby(['Month', 'Category'], as_index=False)['Amount']
        .sum()
        .sort_values(['Month', 'Amount'], ascending=[True, False])
    )

    summary_by_main = (positive_rows
        .groupby(['Month', 'MainCategory'], as_index=False)['Amount']
        .sum()
        .sort_values(['Month', 'Amount'], ascending=[True, False])
    )

    total_positive_by_month = (positive_rows.groupby('Month', as_index=False)['Amount'].sum())

    # Year-level totals
    positive_rows['Year'] = positive_rows['Date'].dt.year
    year_main_totals = (positive_rows
        .groupby(['Year', 'MainCategory'], as_index=False)['Amount']
        .sum()
    )
    subs = (positive_rows
        .groupby(['Year', 'MainCategory'])['Category']
        .apply(lambda s: ', '.join(dict.fromkeys(c for c in s if pd.notna(c))))
        .reset_index(name='Subcategories')
    )
    year_main_totals = year_main_totals.merge(subs, on=['Year', 'MainCategory'], how='left')
    latest_year = int(year_main_totals['Year'].max()) if len(year_main_totals) else None
    latest_year_main_totals = (year_main_totals[year_main_totals['Year'] == latest_year]
        .sort_values('Amount', ascending=False)
    )

    # ---------- JSON-serializable casts ----------
    # Period -> string for Month
    for df in [
        latest_positive_monthly,
        all_positive_monthly,
        summary_by_category,
        summary_by_main,
        total_positive_by_month,
    ]:
        if 'Month' in df.columns:
            df['Month'] = df['Month'].astype(str)

    # Ensure numeric types are plain Python-compatible
    for df, cols in [
        (latest_positive_monthly, ['Amount']),
        (all_positive_monthly, ['Amount']),
        (summary_by_category, ['Amount']),
        (summary_by_main, ['Amount']),
        (total_positive_by_month, ['Amount']),
        (latest_year_main_totals, ['Amount']),
    ]:
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').astype(float)

    # Year as int
    if 'Year' in latest_year_main_totals.columns:
        latest_year_main_totals['Year'] = pd.to_numeric(latest_year_main_totals['Year'], errors='coerce').astype('Int64').astype('int')

    # ---------- return ----------
    return {
        "latest_month": str(latest_month) if latest_month is not pd.NaT else None,
        "positive_monthly": latest_positive_monthly.to_dict(orient='records'),
        "all_positive_monthly": all_positive_monthly.to_dict(orient='records'),
        "summary_by_category": summary_by_category.to_dict(orient='records'),
        "summary_by_main": summary_by_main.to_dict(orient='records'),
        "total_positive_by_month": total_positive_by_month.to_dict(orient='records'),
        "latest_year": latest_year,
        "latest_year_main_totals": latest_year_main_totals.to_dict(orient='records'),
    }

