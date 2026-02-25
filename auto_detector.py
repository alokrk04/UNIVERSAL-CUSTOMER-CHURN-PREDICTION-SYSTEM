"""
auto_detector.py - Universal Dataset Schema Detective
======================================================
Analyses ANY uploaded CSV and figures out:
  - Which column is the customer ID
  - Which column is the churn label (if present)
  - Which columns are dates
  - Which columns are numeric / categorical
  - Whether churn needs to be derived
  - What the best strategy is for this dataset
"""

import pandas as pd
import numpy as np
import re
from typing import Optional

# ── Keyword dictionaries ────────────────────────────────────────────────────────
CHURN_KEYWORDS   = ['churn', 'churned', 'attrition', 'attrited', 'left',
                     'cancel', 'cancelled', 'canceled', 'exit', 'exited',
                     'inactive', 'lapsed', 'dropout', 'unsubscribed']

ID_KEYWORDS      = ['id', 'customer', 'user', 'member', 'account', 'mobile',
                     'phone', 'number', 'cust', 'uid', 'userid', 'client']

DATE_KEYWORDS    = ['date', 'time', 'timestamp', 'created', 'joined', 'signup',
                     'last', 'start', 'end', 'purchase', 'transaction', 'dt']

REVENUE_KEYWORDS = ['amount', 'revenue', 'spend', 'arpu', 'rech', 'recharge',
                     'charge', 'payment', 'price', 'cost', 'value', 'total',
                     'sales', 'income', 'earnings']

USAGE_KEYWORDS   = ['usage', 'mou', 'calls', 'minutes', 'sessions', 'visits',
                     'clicks', 'logins', 'activity', 'frequency', 'vol', 'mb',
                     'gb', 'data', 'transactions', 'orders', 'purchases']


def _col_matches(col: str, keywords: list) -> bool:
    col_lower = col.lower()
    return any(kw in col_lower for kw in keywords)


class SchemaDetector:
    """
    Analyses a DataFrame and returns a schema report describing
    what each column likely represents.
    """

    def __init__(self, df: pd.DataFrame):
        self.df      = df
        self.schema  = {}
        self._detect()

    def _detect(self):
        df = self.df
        self.schema = {
            "churn_col":      self._find_churn_col(),
            "id_col":         self._find_id_col(),
            "date_cols":      self._find_date_cols(),
            "numeric_cols":   self._find_numeric_cols(),
            "categorical_cols": self._find_categorical_cols(),
            "revenue_cols":   self._find_cols_by_kw(REVENUE_KEYWORDS),
            "usage_cols":     self._find_cols_by_kw(USAGE_KEYWORDS),
            "churn_strategy": None,   # filled below
            "n_rows":         len(df),
            "n_cols":         len(df.columns),
        }
        self.schema["churn_strategy"] = self._determine_strategy()

    def _find_churn_col(self) -> Optional[str]:
        """Find an existing churn/label column."""
        df = self.df
        for col in df.columns:
            if _col_matches(col, CHURN_KEYWORDS):
                vals = df[col].dropna().unique()
                # Binary (0/1 or True/False or Yes/No)
                if len(vals) <= 3:
                    return col
        # Also look for binary columns with churn-like distribution
        for col in df.select_dtypes(include=[np.number]).columns:
            if _col_matches(col, CHURN_KEYWORDS):
                return col
        return None

    def _find_id_col(self) -> Optional[str]:
        df = self.df
        candidates = []
        for col in df.columns:
            if _col_matches(col, ID_KEYWORDS):
                # High cardinality = good ID
                if df[col].nunique() > len(df) * 0.5:
                    candidates.append(col)
        if candidates:
            return candidates[0]
        # Fallback: first column with very high cardinality
        for col in df.columns:
            if df[col].nunique() > len(df) * 0.8:
                return col
        return None

    def _find_date_cols(self) -> list:
        df = self.df
        date_cols = []
        # Only check string/object columns — numeric columns are never dates
        str_cols = df.select_dtypes(include=["object"]).columns
        for col in str_cols:
            if _col_matches(col, DATE_KEYWORDS):
                try:
                    sample = df[col].dropna().head(50)
                    converted = pd.to_datetime(sample, errors="coerce")
                    if converted.notna().mean() > 0.7:
                        # Skip uniform columns (all same value)
                        if df[col].nunique() > 5:
                            date_cols.append(col)
                except Exception:
                    pass
        return date_cols

    def _find_numeric_cols(self) -> list:
        df    = self.df
        churn = self.schema.get("churn_col")
        id_c  = self.schema.get("id_col")
        cols  = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude ID and churn col from features
        return [c for c in cols if c not in [churn, id_c]]

    def _find_categorical_cols(self) -> list:
        df    = self.df
        churn = self.schema.get("churn_col")
        id_c  = self.schema.get("id_col")
        date_cols = self.schema.get("date_cols", [])
        cols  = df.select_dtypes(include=['object','category']).columns.tolist()
        return [c for c in cols
                if c not in [churn, id_c] + date_cols
                and df[c].nunique() < 50]

    def _find_cols_by_kw(self, keywords: list) -> list:
        return [c for c in self.df.columns if _col_matches(c, keywords)]

    def _determine_strategy(self) -> dict:
        """
        Decide how churn will be identified/derived for this dataset.
        Returns a strategy dict consumed by the preprocessor.
        """
        s = self.schema
        date_cols = s["date_cols"]
        churn_col = s["churn_col"]

        if churn_col is not None:
            df   = self.df
            vals = df[churn_col].dropna().unique()
            return {
                "type":      "label_exists",
                "churn_col": churn_col,
                "positive":  self._infer_positive_class(df[churn_col])
            }

        # Only use recency if we have a truly varied date column
        varied_dates = []
        for dc in date_cols:
            try:
                parsed = pd.to_datetime(self.df[dc], errors='coerce').dropna()
                if parsed.nunique() > 10:   # must have real variation
                    varied_dates.append(dc)
            except Exception:
                pass

        if varied_dates:
            best_date = self._best_date_col(varied_dates)
            return {
                "type":      "derive_from_recency",
                "date_col":  best_date,
                "percentile": 25,
            }

        # No varied dates — derive from numeric inactivity
        usage_num = [c for c in s["usage_cols"] if c in s["numeric_cols"]]
        if usage_num:
            return {
                "type":       "derive_from_inactivity",
                "usage_cols": usage_num[:5],
            }

        # Last resort: derive from low revenue / spend
        rev_num = [c for c in s["revenue_cols"] if c in s["numeric_cols"]]
        if rev_num:
            return {
                "type":      "derive_from_low_value",
                "rev_cols":  rev_num[:3],
            }

        return {"type": "derive_from_outliers"}

    def _infer_positive_class(self, series: pd.Series):
        """Determine which value means 'churned'."""
        vals = series.dropna().unique()
        for v in vals:
            sv = str(v).lower().strip()
            if sv in ['1', 'yes', 'true', 'y', 'churn', 'churned',
                       'attrited customer', 'exited', 'left']:
                return v
        # Assume minority class is churn
        vc = series.value_counts()
        return vc.index[-1]

    def _best_date_col(self, date_cols: list) -> str:
        """Prefer 'last activity' date over signup date."""
        for kw in ['last', 'recent', 'activity', 'purchase', 'transaction',
                    'visit', 'login']:
            for col in date_cols:
                if kw in col.lower():
                    return col
        return date_cols[0]

    def report(self) -> str:
        s = self.schema
        lines = [
            "\n" + "="*60,
            "  DATASET SCHEMA ANALYSIS",
            "="*60,
            f"  Rows             : {s['n_rows']:,}",
            f"  Columns          : {s['n_cols']}",
            f"  Customer ID col  : {s['id_col'] or 'Not found'}",
            f"  Churn label col  : {s['churn_col'] or 'Not found — will derive'}",
            f"  Date columns     : {s['date_cols'] or 'None'}",
            f"  Numeric features : {len(s['numeric_cols'])}",
            f"  Categorical feat : {len(s['categorical_cols'])}",
            f"  Revenue columns  : {s['revenue_cols'][:4]}",
            f"  Usage columns    : {s['usage_cols'][:4]}",
            f"  Churn strategy   : {s['churn_strategy']['type']}",
            "="*60,
        ]
        return "\n".join(lines)
