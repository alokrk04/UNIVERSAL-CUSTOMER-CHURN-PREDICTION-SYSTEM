"""
universal_features.py - Adaptive Feature Engineering
=====================================================
Automatically engineers the best possible features
from ANY dataset regardless of column names or domain.

Strategy:
  1. Derive ratio features between related numeric columns
  2. Identify & engineer trend/decline features
  3. Create interaction features for high-importance pairs
  4. Add date-derived temporal features if dates exist
  5. Compute per-row statistical summary features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    result = a / b.replace(0, np.nan)
    result = result.replace([np.inf, -np.inf], np.nan).fillna(0)
    cap = result.abs().quantile(0.99) + 1e-9
    return result.clip(-cap, cap)


def engineer_features(df: pd.DataFrame, schema: dict,
                       df_raw: pd.DataFrame = None) -> pd.DataFrame:
    """
    Adds engineered features to the cleaned DataFrame.
    Returns augmented DataFrame.
    """
    print("[Features] Engineering universal features ...")
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # ── 1. Row-level statistical summary features ──────────────────────────────
    # These work on EVERY dataset regardless of domain
    df['__feat_row_mean']   = df[num_cols].mean(axis=1)
    df['__feat_row_std']    = df[num_cols].std(axis=1).fillna(0)
    df['__feat_row_min']    = df[num_cols].min(axis=1)
    df['__feat_row_max']    = df[num_cols].max(axis=1)
    df['__feat_row_range']  = df['__feat_row_max'] - df['__feat_row_min']
    df['__feat_zero_count'] = (df[num_cols] == 0).sum(axis=1)
    df['__feat_null_count'] = df[num_cols].isna().sum(axis=1)

    # ── 2. Revenue / spend features ────────────────────────────────────────────
    rev_cols = [c for c in schema.get("revenue_cols", []) if c in df.columns]
    if len(rev_cols) >= 2:
        df['__feat_total_revenue'] = df[rev_cols].sum(axis=1)
        df['__feat_avg_revenue']   = df[rev_cols].mean(axis=1)
        df['__feat_revenue_std']   = df[rev_cols].std(axis=1).fillna(0)
        # Trend: last vs first revenue column
        df['__feat_revenue_trend'] = _safe_ratio(
            df[rev_cols[-1]], df[rev_cols[0]])

    elif len(rev_cols) == 1:
        df['__feat_total_revenue'] = df[rev_cols[0]]

    # ── 3. Usage / activity features ──────────────────────────────────────────
    usage_cols = [c for c in schema.get("usage_cols", []) if c in df.columns]
    if len(usage_cols) >= 2:
        df['__feat_total_usage'] = df[usage_cols].sum(axis=1)
        df['__feat_avg_usage']   = df[usage_cols].mean(axis=1)
        df['__feat_usage_std']   = df[usage_cols].std(axis=1).fillna(0)
        df['__feat_usage_trend'] = _safe_ratio(
            df[usage_cols[-1]], df[usage_cols[0]])
        # Activity consistency (low std relative to mean = consistent user)
        df['__feat_usage_cv'] = _safe_ratio(
            df['__feat_usage_std'], df['__feat_avg_usage'] + 1)
    elif len(usage_cols) == 1:
        df['__feat_total_usage'] = df[usage_cols[0]]

    # ── 4. Temporal features from date columns ─────────────────────────────────
    date_cols = schema.get("date_cols", [])
    if date_cols and df_raw is not None:
        for dcol in date_cols[:3]:   # max 3 date columns
            try:
                parsed = pd.to_datetime(df_raw[dcol], errors='coerce')
                obs    = parsed.max()
                feat_name = f"__feat_days_since_{dcol}"
                df[feat_name] = (obs - parsed).dt.days.fillna(
                    (obs - parsed).dt.days.median())

                # Month and day of week as cyclical features
                df[f"__feat_{dcol}_month_sin"] = np.sin(
                    2*np.pi * parsed.dt.month / 12)
                df[f"__feat_{dcol}_month_cos"] = np.cos(
                    2*np.pi * parsed.dt.month / 12)
            except Exception:
                pass

    # ── 5. Ratio features between top numeric columns ─────────────────────────
    # (revenue / usage = value per unit of activity)
    if rev_cols and usage_cols:
        rev_sum   = df[rev_cols].sum(axis=1)   if len(rev_cols)   > 1 else df[rev_cols[0]]
        usage_sum = df[usage_cols].sum(axis=1) if len(usage_cols) > 1 else df[usage_cols[0]]
        df['__feat_revenue_per_usage'] = _safe_ratio(rev_sum, usage_sum)

    # ── 6. High cardinality interaction (top 5 numeric pairs) ─────────────────
    top_cols = num_cols[:5]   # limit to avoid explosion
    for i in range(len(top_cols)):
        for j in range(i+1, len(top_cols)):
            c1, c2 = top_cols[i], top_cols[j]
            df[f'__feat_ratio_{i}_{j}'] = _safe_ratio(df[c1], df[c2] + 1)

    # ── 7. Inactivity score (0 = very active, 1 = completely inactive) ─────────
    all_activity = [c for c in usage_cols + rev_cols if c in df.columns]
    if all_activity:
        act_sum = df[all_activity].sum(axis=1)
        max_act = act_sum.quantile(0.99) + 1
        df['__feat_inactivity_score'] = 1 - (act_sum / max_act).clip(0, 1)

    # Final cleanup: replace any remaining inf/nan with 0
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    new_feats = [c for c in df.columns if c.startswith('__feat')]
    print(f"[Features] Added {len(new_feats)} engineered features | "
          f"Total: {df.shape[1]} columns")
    return df


def select_features(X: pd.DataFrame, y: pd.Series,
                     max_features: int = 150) -> pd.DataFrame:
    """
    If dataset has too many columns, use correlation + variance
    to select the most informative ones. Keeps all engineered features.
    """
    if X.shape[1] <= max_features:
        return X

    print(f"[Features] Selecting top {max_features} from {X.shape[1]} features ...")

    # Keep all engineered features
    eng_cols  = [c for c in X.columns if c.startswith('__feat')]

    # Rank raw features by absolute correlation with y
    raw_cols  = [c for c in X.columns if not c.startswith('__feat')]
    corr      = X[raw_cols].corrwith(y).abs().sort_values(ascending=False)
    top_raw   = corr.head(max_features - len(eng_cols)).index.tolist()

    selected  = list(set(eng_cols + top_raw))
    print(f"[Features] Selected {len(selected)} features")
    return X[selected]
