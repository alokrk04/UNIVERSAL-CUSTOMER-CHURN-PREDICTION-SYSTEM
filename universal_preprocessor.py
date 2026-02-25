"""
universal_preprocessor.py - Adaptive ETL for Any Dataset
==========================================================
Works with ANY CSV file. Handles:
  - Existing churn labels (binary, Yes/No, True/False)
  - Derived churn from recency / inactivity / low spend
  - Missing values, outliers, encoding
  - Train/val/test split
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TEST_SIZE, VAL_SIZE, RANDOM_STATE
from auto_detector import SchemaDetector


# ── Loading ────────────────────────────────────────────────────────────────────

def load_any_csv(path: str) -> pd.DataFrame:
    """Load CSV with smart encoding detection."""
    for enc in ['utf-8', 'latin-1', 'cp1252', 'utf-16']:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            print(f"[ETL] Loaded '{os.path.basename(path)}' "
                  f"({len(df):,} rows × {len(df.columns)} cols) [{enc}]")
            return df
        except Exception:
            continue
    raise ValueError(f"Could not read {path} with any known encoding.")


# ── Churn Label Derivation ─────────────────────────────────────────────────────

def derive_churn(df: pd.DataFrame, strategy: dict) -> pd.Series:
    """Apply the appropriate churn derivation strategy."""
    stype = strategy["type"]

    if stype == "label_exists":
        col      = strategy["churn_col"]
        positive = strategy["positive"]
        y = (df[col] == positive).astype(int)
        print(f"[ETL] Churn label from column '{col}' "
              f"(positive class = '{positive}')")

    elif stype == "derive_from_recency":
        col      = strategy["date_col"]
        pct      = strategy["percentile"]
        dates    = pd.to_datetime(df[col], errors='coerce')
        obs_date = dates.max()
        recency  = (obs_date - dates).dt.days.fillna(9999)
        threshold = np.percentile(recency.replace(9999, np.nan).dropna(), 100 - pct)
        y = (recency >= threshold).astype(int)
        print(f"[ETL] Churn derived from recency '{col}' "
              f"(> {threshold:.0f} days = churned)")

    elif stype == "derive_from_inactivity":
        cols = strategy["usage_cols"]
        # Zero or very low usage across all usage columns = churned
        usage_sum = df[cols].fillna(0).sum(axis=1)
        threshold = usage_sum.quantile(0.20)
        y = (usage_sum <= threshold).astype(int)
        print(f"[ETL] Churn derived from inactivity "
              f"(total usage ≤ {threshold:.2f} = churned)")

    elif stype == "derive_from_low_value":
        cols = strategy["rev_cols"]
        rev_sum   = df[cols].fillna(0).sum(axis=1)
        threshold = rev_sum.quantile(0.20)
        y = (rev_sum <= threshold).astype(int)
        print(f"[ETL] Churn derived from low value "
              f"(revenue ≤ {threshold:.2f} = churned)")

    else:  # derive_from_outliers
        num_df    = df.select_dtypes(include=[np.number]).fillna(0)
        row_means = num_df.mean(axis=1)
        threshold = row_means.quantile(0.20)
        y = (row_means <= threshold).astype(int)
        print(f"[ETL] Churn derived from low overall activity")

    churn_rate = y.mean()
    print(f"[ETL] Churn rate: {churn_rate:.2%}  "
          f"({y.sum():,} churned / {len(y):,} total)")

    # Auto-recover: if churn rate is unrealistic, try alternate strategies
    if churn_rate < 0.01 or churn_rate > 0.90:
        print(f"[WARNING] Churn rate {churn_rate:.1%} is unrealistic. "
              f"Trying alternate derivation ...")
        y = _derive_churn_fallback(df, strategy)
        churn_rate = y.mean()
        print(f"[ETL] Revised churn rate: {churn_rate:.2%}  "
              f"({y.sum():,} churned / {len(y):,} total)")
    return y


def _derive_churn_fallback(df: pd.DataFrame, failed_strategy: dict) -> pd.Series:
    """
    If the primary churn strategy gives an unrealistic rate,
    try each fallback in order until we get a sensible rate (5-50%).
    """
    # Strategy 1: use numeric columns with 'churn' in name as probability
    for col in df.columns:
        clow = col.lower()
        if any(k in clow for k in ['churn','attrition','exit','left']):
            if df[col].dtype in ['float64','int64']:
                vals = df[col].dropna()
                if vals.nunique() == 2:
                    y = (df[col] == vals.max()).astype(int)
                    if 0.03 <= y.mean() <= 0.70:
                        print(f"[ETL-Fallback] Used '{col}' as churn label")
                        return y

    # Strategy 2: bottom 20% of total numeric activity = churned
    num_df = df.select_dtypes(include=[np.number])
    # Exclude ID-like columns (high cardinality integers)
    feature_cols = [c for c in num_df.columns
                    if num_df[c].nunique() < len(df) * 0.9]
    if feature_cols:
        activity = num_df[feature_cols].fillna(0).sum(axis=1)
        threshold = activity.quantile(0.20)
        y = (activity <= threshold).astype(int)
        if 0.05 <= y.mean() <= 0.50:
            print(f"[ETL-Fallback] Derived churn from bottom-20% activity")
            return y

    # Strategy 3: force a 15% churn rate using lowest activity scores
    num_df2 = df.select_dtypes(include=[np.number]).fillna(0)
    row_score = num_df2.sum(axis=1)
    threshold = row_score.quantile(0.15)
    y = (row_score <= threshold).astype(int)
    print(f"[ETL-Fallback] Forced 15% churn from lowest activity rows")
    return y


# ── Cleaning ───────────────────────────────────────────────────────────────────

def clean_dataframe(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """
    Universal cleaning:
      1. Remove leakage (churn col itself from features)
      2. Drop useless columns (all-null, single-value, pure IDs)
      3. Handle missing values
      4. Clip numeric outliers
      5. Encode categoricals
    """
    print("[ETL] Cleaning ...")
    df = df.copy()

    # Remove churn label and ID from features
    drop_always = [c for c in [schema.get("churn_col"), schema.get("id_col")]
                   if c is not None]
    df = df.drop(columns=drop_always, errors='ignore')

    # Remove date columns (encoded separately if needed)
    df = df.drop(columns=schema.get("date_cols", []), errors='ignore')

    # Drop columns that are all-null or single-value
    null_cols  = [c for c in df.columns if df[c].isna().all()]
    const_cols = [c for c in df.columns if df[c].nunique() <= 1]
    df = df.drop(columns=null_cols + const_cols)

    # Drop very high cardinality object columns (free text / IDs)
    obj_cols = df.select_dtypes(include='object').columns
    high_card = [c for c in obj_cols if df[c].nunique() > 50]
    df = df.drop(columns=high_card)

    # Encode remaining categoricals
    cat_cols = df.select_dtypes(include=['object','category','bool']).columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = df[col].astype(str).fillna('Unknown')
        try:
            df[col] = le.fit_transform(df[col])
        except Exception:
            df[col] = pd.factorize(df[col])[0]

    # Fill numeric NaN with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Clip numeric outliers (1st–99th percentile)
    for col in num_cols:
        lo = df[col].quantile(0.01)
        hi = df[col].quantile(0.99)
        df[col] = df[col].clip(lo, hi)

    print(f"[ETL] Clean shape: {df.shape}")
    return df


# ── Split ──────────────────────────────────────────────────────────────────────

def split_dataset(X, y):
    churn_count = int(y.sum())
    total       = len(y)

    # If churn is too rare (< 10 cases), fall back to no stratification
    # and warn the user their churn derivation needs review
    use_stratify = churn_count >= 10

    if not use_stratify:
        print(f"[WARNING] Only {churn_count} churned customer(s) found!")
        print("[WARNING] Churn derivation may be incorrect for this dataset.")
        print("[WARNING] Falling back to non-stratified split.")

    def _split(X, y, test_size, stratify):
        return train_test_split(X, y, test_size=test_size,
                                stratify=stratify, random_state=RANDOM_STATE)

    strat = y if use_stratify else None
    X_tv, X_test, y_tv, y_test = _split(X, y, TEST_SIZE, strat)
    strat2 = y_tv if use_stratify else None
    X_train, X_val, y_train, y_val = _split(
        X_tv, y_tv, VAL_SIZE/(1-TEST_SIZE), strat2)

    print(f"[Split] Train:{len(X_train):,}  "
          f"Val:{len(X_val):,}  Test:{len(X_test):,}")
    print(f"[Split] Churn in test set: "
          f"{int(y_test.sum()):,} / {len(y_test):,} ({y_test.mean():.1%})")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Master Pipeline ─────────────────────────────────────────────────────────────

def run_universal_etl(csv_path: str):
    """
    Full ETL for any CSV.
    Returns: X (features), y (churn), df_raw, schema
    """
    df_raw   = load_any_csv(csv_path)
    detector = SchemaDetector(df_raw)
    print(detector.report())

    schema   = detector.schema
    strategy = schema["churn_strategy"]

    y        = derive_churn(df_raw, strategy)
    df_clean = clean_dataframe(df_raw, schema)

    # Align index
    y = y.reset_index(drop=True)
    df_clean = df_clean.reset_index(drop=True)

    return df_clean, y, df_raw, schema


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/telecom_churn_data.csv"
    X, y, df_raw, schema = run_universal_etl(path)
    print(f"\nFeature matrix: {X.shape}")
    print(f"Churn rate: {y.mean():.2%}")