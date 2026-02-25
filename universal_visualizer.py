"""
universal_visualizer.py  —  Fully Data-Driven Chart Engine
===========================================================
NO hardcoded column names.  NO assumed chart types.

How it works:
  1. DataProfiler  scans the dataset and builds a profile
     (which cols are numeric, categorical, temporal, high-variance, etc.)
  2. ChartPlanner  decides WHICH chart types to generate based on
     what the profile actually contains  (skips charts that don't apply)
  3. ChartEngine   draws each planned chart using the real column names
     and labels everything with the dataset's own terminology

Every chart title, axis label, and legend entry is derived from the
actual data — nothing is ever hardcoded.
"""

import os, sys, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUT_DIR

# ── Global aesthetics ──────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="Set2")
PALETTE   = sns.color_palette("Set2")
C_CHURN   = "#e74c3c"
C_RETAIN  = "#27ae60"
C_NEUTRAL = "#3498db"
C_BANDS   = ["#27ae60", "#f39c12", "#e67e22", "#e74c3c"]

plt.rcParams.update({
    "figure.dpi":       150,
    "figure.facecolor": "white",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "legend.fontsize":  9,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA PROFILER
# ══════════════════════════════════════════════════════════════════════════════

class DataProfiler:
    """
    Scans the raw DataFrame and produces a rich profile that the
    ChartPlanner uses to decide which charts are applicable.
    """

    def __init__(self, df_raw: pd.DataFrame, df_feat: pd.DataFrame,
                 y: pd.Series, schema: dict):
        self.df_raw  = df_raw.copy().reset_index(drop=True)
        self.df_feat = df_feat.copy().reset_index(drop=True)
        self.y       = y.reset_index(drop=True)
        self.schema  = schema
        self.profile = {}
        self._build()

    # ── internal helpers ──────────────────────────────────────────────────────

    def _label(self, col):
        """Human-readable label from a column name."""
        return col.replace("_", " ").replace("-", " ").title()

    def _top_corr_cols(self, n=20):
        """Top-n numeric columns by absolute correlation with churn label."""
        num = self.df_feat.select_dtypes(include=[np.number])
        corr = num.corrwith(self.y).abs().sort_values(ascending=False)
        return corr.head(n).index.tolist()

    def _varied_num_cols(self, df, n=30):
        """Numeric columns with meaningful variance (excludes near-constant)."""
        num = df.select_dtypes(include=[np.number])
        std = num.std()
        return std[std > 0].sort_values(ascending=False).head(n).index.tolist()

    def _parse_date_col(self, col):
        """Try to parse a column as dates; return Series or None."""
        try:
            parsed = pd.to_datetime(self.df_raw[col], errors="coerce")
            if parsed.notna().mean() > 0.7 and parsed.nunique() > 5:
                return parsed
        except Exception:
            pass
        return None

    # ── build profile ─────────────────────────────────────────────────────────

    def _build(self):
        p = {}

        # ── basic counts ──────────────────────────────────────────────────────
        p["n_rows"]     = len(self.df_raw)
        p["n_churned"]  = int(self.y.sum())
        p["n_retained"] = int((1 - self.y).sum())
        p["churn_rate"] = float(self.y.mean())

        # ── numeric columns (from clean feature matrix) ────────────────────────
        p["top_corr_cols"]   = self._top_corr_cols(20)
        p["varied_num_cols"] = self._varied_num_cols(self.df_feat, 30)

        # ── revenue columns ────────────────────────────────────────────────────
        rev = [c for c in self.schema.get("revenue_cols", [])
               if c in self.df_raw.columns
               and pd.api.types.is_numeric_dtype(self.df_raw[c])]
        p["revenue_cols"] = rev[:8]

        # ── usage / activity columns ───────────────────────────────────────────
        usage = [c for c in self.schema.get("usage_cols", [])
                 if c in self.df_raw.columns
                 and pd.api.types.is_numeric_dtype(self.df_raw[c])
                 and self.df_raw[c].nunique() > 2]
        p["usage_cols"] = usage[:8]

        # ── categorical columns (raw, low-cardinality) ─────────────────────────
        cat = []
        for col in self.schema.get("categorical_cols", []):
            if col not in self.df_raw.columns:
                continue
            n_uniq = self.df_raw[col].nunique()
            if 2 <= n_uniq <= 20:
                cat.append(col)
        p["cat_cols"] = cat[:5]

        # ── temporal columns ──────────────────────────────────────────────────
        date_series = {}
        for col in self.schema.get("date_cols", []):
            parsed = self._parse_date_col(col)
            if parsed is not None:
                date_series[col] = parsed
        p["date_series"]  = date_series          # {col_name: parsed_Series}
        p["has_dates"]    = len(date_series) > 0

        # ── multi-period columns ───────────────────────────────────────────────
        # Detect columns that represent the same metric across periods
        # e.g. arpu_6, arpu_7, arpu_8  or  revenue_q1, revenue_q2 …
        period_groups = self._find_period_groups()
        p["period_groups"] = period_groups       # {base_name: [col1, col2, …]}

        # ── binary / flag columns ─────────────────────────────────────────────
        bin_cols = [c for c in self.df_raw.select_dtypes(include=[np.number]).columns
                    if self.df_raw[c].dropna().isin([0, 1]).all()
                    and c not in [self.schema.get("churn_col")]
                    and self.df_raw[c].nunique() == 2]
        p["binary_cols"] = bin_cols[:6]

        self.profile = p

    def _find_period_groups(self):
        """
        Detect column families like [arpu_6, arpu_7, arpu_8] or
        [sales_q1, sales_q2, sales_q3] by stripping trailing digits/tokens.
        """
        import re
        cols = self.df_raw.select_dtypes(include=[np.number]).columns.tolist()
        groups = {}
        for col in cols:
            base = re.sub(r'[_\-]?\d+$', '', col).strip('_- ')
            if base and base != col:
                groups.setdefault(base, []).append(col)
        # Keep groups with 2–12 members (genuine multi-period)
        return {k: sorted(v) for k, v in groups.items()
                if 2 <= len(v) <= 12}


# ══════════════════════════════════════════════════════════════════════════════
# 2.  CHART PLANNER
# ══════════════════════════════════════════════════════════════════════════════

class ChartPlanner:
    """
    Examines the DataProfiler output and returns an ordered list of
    (chart_id, params) tuples — one per chart that makes sense to draw.
    Charts that have no applicable data are silently skipped.
    """

    def __init__(self, profiler: DataProfiler):
        self.p = profiler.profile
        self.schema = profiler.schema

    def plan(self):
        plan = []
        idx  = 1

        # Always generated (always have y and churn_probs)
        plan.append((idx, "churn_overview",    {})); idx += 1
        plan.append((idx, "risk_segments",     {})); idx += 1
        plan.append((idx, "risk_distribution", {})); idx += 1

        # Feature importance — always available after model training
        plan.append((idx, "feature_importance", {})); idx += 1

        # Model comparison — always available
        plan.append((idx, "model_comparison", {})); idx += 1

        # Top correlated features distribution (KDE / hist)
        if self.p["top_corr_cols"]:
            plan.append((idx, "top_feature_distributions",
                          {"cols": self.p["top_corr_cols"][:9]})); idx += 1

        # Mean comparison bar chart (churned vs retained)
        if self.p["varied_num_cols"]:
            plan.append((idx, "mean_comparison",
                          {"cols": self.p["varied_num_cols"][:12]})); idx += 1

        # Correlation heatmap
        if len(self.p["top_corr_cols"]) >= 4:
            plan.append((idx, "correlation_heatmap",
                          {"cols": self.p["top_corr_cols"][:15]})); idx += 1

        # Multi-period trend lines (one chart per group, max 3 groups)
        for base, cols in list(self.p["period_groups"].items())[:3]:
            plan.append((idx, "period_trend",
                          {"base": base, "cols": cols})); idx += 1

        # Revenue columns comparison (if present and not already in period groups)
        non_period_rev = [c for c in self.p["revenue_cols"]
                           if not any(c in v for v in
                                       self.p["period_groups"].values())]
        if non_period_rev:
            plan.append((idx, "revenue_comparison",
                          {"cols": non_period_rev})); idx += 1

        # Usage columns comparison
        non_period_usage = [c for c in self.p["usage_cols"]
                             if not any(c in v for v in
                                         self.p["period_groups"].values())]
        if non_period_usage:
            plan.append((idx, "usage_comparison",
                          {"cols": non_period_usage})); idx += 1

        # Categorical churn rates
        if self.p["cat_cols"]:
            plan.append((idx, "churn_by_category",
                          {"cols": self.p["cat_cols"]})); idx += 1

        # Binary flag analysis
        if self.p["binary_cols"]:
            plan.append((idx, "binary_flag_rates",
                          {"cols": self.p["binary_cols"]})); idx += 1

        # Temporal trend (if valid date + a metric column)
        if self.p["has_dates"]:
            metric_col = (self.p["revenue_cols"] or
                           self.p["usage_cols"] or
                           self.p["varied_num_cols"])
            if metric_col:
                plan.append((idx, "temporal_trend",
                              {"metric_col": metric_col[0]})); idx += 1

        return plan


# ══════════════════════════════════════════════════════════════════════════════
# 3.  CHART ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class ChartEngine:
    """
    Receives the chart plan and draws each chart using real column names.
    """

    def __init__(self, profiler: DataProfiler,
                 churn_probs: np.ndarray,
                 importance_df: pd.DataFrame,
                 results: list,
                 dataset_name: str):
        self.prof         = profiler
        self.p            = profiler.profile
        self.df_raw       = profiler.df_raw
        self.df_feat      = profiler.df_feat
        self.y            = profiler.y
        self.churn_probs  = churn_probs
        self.importance_df= importance_df
        self.results      = results
        self.dataset_name = dataset_name
        self.generated    = []

    # ── save helper ────────────────────────────────────────────────────────────

    def _save(self, fig, idx: int, slug: str):
        filename = f"{idx:02d}_{slug}.png"
        path     = os.path.join(OUTPUT_DIR, filename)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{idx:02d}] {filename}")
        self.generated.append(path)
        return path

    def _label(self, col):
        return col.replace("_", " ").replace("-", " ").title()

    def _df_with_churn(self, df):
        df2 = df.copy()
        df2["__churn__"] = self.y.values
        return df2

    # ── chart renderers ────────────────────────────────────────────────────────

    def draw_churn_overview(self, idx):
        p   = self.p
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Churn Overview — {self.dataset_name}",
                     fontweight="bold", fontsize=14)

        sizes  = [p["n_retained"], p["n_churned"]]
        labels = [f"Retained\n{p['n_retained']:,}", f"Churned\n{p['n_churned']:,}"]
        ax1.pie(sizes, labels=labels, colors=[C_RETAIN, C_CHURN],
                autopct="%1.1f%%", startangle=90,
                wedgeprops=dict(edgecolor="white", linewidth=2))
        ax1.set_title("Churn vs Retained Split")

        ax2.bar(["Retained", "Churned"], sizes, color=[C_RETAIN, C_CHURN],
                edgecolor="white", width=0.45)
        for i, v in enumerate(sizes):
            ax2.text(i, v + p["n_rows"] * 0.005,
                     f"{v:,}\n({v/p['n_rows']:.1%})",
                     ha="center", fontweight="bold")
        ax2.set_title("Customer Count"); ax2.set_ylabel("Customers")
        ax2.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax2.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        self._save(fig, idx, "churn_overview")

    def draw_risk_segments(self, idx):
        thresholds = [0, 0.30, 0.60, 0.80, 1.01]
        labels = ["🟢 Low\n(0-30%)", "🟡 Medium\n(30-60%)",
                   "🟠 High\n(60-80%)", "🔴 Critical\n(80-100%)"]
        counts = [((self.churn_probs >= thresholds[i]) &
                    (self.churn_probs < thresholds[i+1])).sum()
                   for i in range(4)]
        total = len(self.churn_probs)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        ax1.bar(labels, counts, color=C_BANDS, edgecolor="white", width=0.5)
        for i, v in enumerate(counts):
            ax1.text(i, v + total * 0.005,
                     f"{v:,}\n({v/total:.1%})",
                     ha="center", fontsize=9, fontweight="bold")
        ax1.set_title("Customer Risk Segment Counts", fontweight="bold")
        ax1.set_ylabel("Customers"); ax1.grid(axis="y", alpha=0.3)

        ax2.pie(counts, labels=labels, colors=C_BANDS, autopct="%1.1f%%",
                startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2))
        ax2.set_title("Risk Segment Distribution", fontweight="bold")
        fig.suptitle(f"Churn Risk Segmentation — {self.dataset_name}",
                     fontweight="bold", fontsize=14)
        plt.tight_layout()
        self._save(fig, idx, "risk_segments")

    def draw_risk_distribution(self, idx):
        probs = self.churn_probs
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(probs, bins=50, color=PALETTE[0], alpha=0.6,
                edgecolor="white", density=True)
        try:
            sns.kdeplot(probs, ax=ax, color=PALETTE[1], linewidth=2.5)
        except Exception:
            pass
        ax.axvline(0.5, color="red", linestyle="--",
                   linewidth=2, label="Risk Threshold (50%)")
        high = (probs >= 0.5).mean() * 100
        ax.text(0.52, ax.get_ylim()[1] * 0.85,
                f"High Risk\n{high:.1f}%", color="red",
                fontsize=10, fontweight="bold")
        ax.set_title(f"Predicted Churn Probability Distribution — {self.dataset_name}",
                     fontweight="bold")
        ax.set_xlabel("Churn Probability"); ax.set_ylabel("Density")
        ax.legend()
        plt.tight_layout()
        self._save(fig, idx, "risk_distribution")

    def draw_feature_importance(self, idx):
        df  = self.importance_df.head(20)
        fig, ax = plt.subplots(figsize=(10, 7))
        colors  = [C_CHURN if i < 5 else PALETTE[2] for i in range(len(df))]
        ax.barh([self._label(f) for f in df["feature"]][::-1],
                df["importance"][::-1],
                color=colors[::-1], edgecolor="white")
        ax.set_title(f"Top {len(df)} Churn Predictors — {self.dataset_name}",
                     fontweight="bold")
        ax.set_xlabel("Feature Importance Score")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        self._save(fig, idx, "feature_importance")

    def draw_model_comparison(self, idx):
        if not self.results:
            return
        df      = pd.DataFrame(self.results)
        metrics = [m for m in ["roc_auc", "f1", "precision", "recall", "accuracy"]
                   if m in df.columns]
        x, w    = np.arange(len(df)), 0.15
        fig, ax = plt.subplots(figsize=(11, 5))
        for i, m in enumerate(metrics):
            ax.bar(x + i * w, df[m], w,
                   label=m.replace("_", " ").upper(),
                   color=PALETTE[i % len(PALETTE)], edgecolor="white")
        ax.set_xticks(x + w * (len(metrics) - 1) / 2)
        ax.set_xticklabels(df["model"])
        ax.set_ylim(0, 1.12); ax.set_ylabel("Score")
        ax.set_title("Model Performance Comparison", fontweight="bold")
        ax.legend(loc="lower right"); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        self._save(fig, idx, "model_comparison")

    def draw_top_feature_distributions(self, idx, cols):
        """KDE / histogram of top correlated features, split by churn."""
        cols    = [c for c in cols if c in self.df_feat.columns][:9]
        if not cols:
            return
        df2     = self._df_with_churn(self.df_feat)
        nrows   = (len(cols) + 2) // 3
        fig, axes = plt.subplots(nrows, 3, figsize=(15, 4 * nrows))
        axes    = np.array(axes).flatten()

        for i, col in enumerate(cols):
            ax = axes[i]
            p1, p99 = np.percentile(df2[col].dropna(), [1, 99])
            for val, color, label in [(0, C_RETAIN, "Retained"),
                                       (1, C_CHURN,  "Churned")]:
                data = df2[df2["__churn__"] == val][col].clip(p1, p99)
                try:
                    sns.kdeplot(data, ax=ax, color=color, linewidth=2,
                                fill=True, alpha=0.3, label=label)
                except Exception:
                    ax.hist(data, bins=25, alpha=0.5,
                            color=color, label=label, density=True)
            ax.set_title(self._label(col), fontsize=10)
            ax.set_xlabel(""); ax.legend(fontsize=8); ax.grid(alpha=0.3)

        for j in range(len(cols), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            f"Top Feature Distributions: Churned vs Retained — {self.dataset_name}",
            fontweight="bold", fontsize=13, y=1.01)
        plt.tight_layout()
        self._save(fig, idx, "feature_distributions")

    def draw_mean_comparison(self, idx, cols):
        """Grouped bar: normalised mean per feature for churned vs retained."""
        cols    = [c for c in cols if c in self.df_feat.columns][:12]
        if not cols:
            return
        df2     = self._df_with_churn(self.df_feat[cols])
        means   = df2.groupby("__churn__")[cols].mean()
        # Normalise to 0-1 for cross-column comparability
        norm    = (means - means.min()) / (means.max() - means.min() + 1e-9)

        x, w    = np.arange(len(cols)), 0.35
        fig, ax = plt.subplots(figsize=(max(12, len(cols)), 5))
        if 0 in norm.index:
            ax.bar(x - w/2, norm.loc[0].values, w,
                   label="Retained", color=C_RETAIN, alpha=0.85)
        if 1 in norm.index:
            ax.bar(x + w/2, norm.loc[1].values, w,
                   label="Churned", color=C_CHURN, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([self._label(c) for c in cols],
                            rotation=35, ha="right", fontsize=9)
        ax.set_title(
            f"Key Metrics Comparison: Churned vs Retained (Normalised) — {self.dataset_name}",
            fontweight="bold")
        ax.set_ylabel("Normalised Mean (0=low, 1=high)")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        self._save(fig, idx, "mean_comparison")

    def draw_correlation_heatmap(self, idx, cols):
        cols = [c for c in cols if c in self.df_feat.columns]
        if len(cols) < 4:
            return
        corr = self.df_feat[cols].corr()
        fig, ax = plt.subplots(figsize=(max(10, len(cols)), max(8, len(cols))))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, ax=ax, mask=mask, cmap="RdYlGn",
                    center=0, vmin=-1, vmax=1, annot=len(cols) <= 15,
                    fmt=".2f", annot_kws={"size": 7},
                    linewidths=0.5, square=True,
                    cbar_kws={"shrink": 0.8})
        ax.set_xticklabels([self._label(c) for c in corr.columns],
                            rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels([self._label(c) for c in corr.index],
                            rotation=0, fontsize=8)
        ax.set_title(
            f"Feature Correlation Heatmap — {self.dataset_name}",
            fontweight="bold", pad=20)
        plt.tight_layout()
        self._save(fig, idx, "correlation_heatmap")

    def draw_period_trend(self, idx, base, cols):
        """
        For column families like [arpu_6, arpu_7, arpu_8]:
        line chart of mean per period, split by churn.
        """
        cols    = [c for c in cols if c in self.df_raw.columns
                   and pd.api.types.is_numeric_dtype(self.df_raw[c])]
        if len(cols) < 2:
            return
        df2     = self._df_with_churn(self.df_raw[cols])
        means   = df2.groupby("__churn__")[cols].mean()

        # Try to extract period labels from column names
        import re
        labels  = [re.sub(r'^.*?[_\-](\w+)$', r'\1', c) for c in cols]

        fig, ax = plt.subplots(figsize=(10, 5))
        if 0 in means.index:
            ax.plot(labels, means.loc[0].values, marker="o", linewidth=2.5,
                    color=C_RETAIN, label="Retained", markersize=7)
        if 1 in means.index:
            ax.plot(labels, means.loc[1].values, marker="s", linewidth=2.5,
                    linestyle="--", color=C_CHURN, label="Churned", markersize=7)

        ax.set_title(
            f"{self._label(base)} Trend Across Periods: Churned vs Retained — {self.dataset_name}",
            fontweight="bold")
        ax.set_xlabel("Period"); ax.set_ylabel(f"Mean {self._label(base)}")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        slug = f"period_trend_{base[:20].replace(' ','_')}"
        self._save(fig, idx, slug)

    def draw_revenue_comparison(self, idx, cols):
        """Bar chart of revenue columns, churned vs retained."""
        cols = [c for c in cols if c in self.df_raw.columns
                and pd.api.types.is_numeric_dtype(self.df_raw[c])][:8]
        if not cols:
            return
        df2   = self._df_with_churn(self.df_raw[cols])
        means = df2.groupby("__churn__")[cols].mean()

        x, w  = np.arange(len(cols)), 0.35
        fig, ax = plt.subplots(figsize=(max(10, len(cols) * 1.5), 5))
        if 0 in means.index:
            ax.bar(x - w/2, means.loc[0].values, w,
                   label="Retained", color=C_RETAIN, alpha=0.85)
        if 1 in means.index:
            ax.bar(x + w/2, means.loc[1].values, w,
                   label="Churned", color=C_CHURN, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([self._label(c) for c in cols],
                            rotation=30, ha="right", fontsize=9)
        ax.set_title(
            f"Revenue Metrics: Churned vs Retained — {self.dataset_name}",
            fontweight="bold")
        ax.set_ylabel("Mean Value"); ax.legend(); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        self._save(fig, idx, "revenue_comparison")

    def draw_usage_comparison(self, idx, cols):
        """Bar chart of usage columns, churned vs retained."""
        cols = [c for c in cols if c in self.df_raw.columns
                and pd.api.types.is_numeric_dtype(self.df_raw[c])][:8]
        if not cols:
            return
        df2   = self._df_with_churn(self.df_raw[cols])
        means = df2.groupby("__churn__")[cols].mean()

        x, w  = np.arange(len(cols)), 0.35
        fig, ax = plt.subplots(figsize=(max(10, len(cols) * 1.5), 5))
        if 0 in means.index:
            ax.bar(x - w/2, means.loc[0].values, w,
                   label="Retained", color=C_RETAIN, alpha=0.85)
        if 1 in means.index:
            ax.bar(x + w/2, means.loc[1].values, w,
                   label="Churned", color=C_CHURN, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([self._label(c) for c in cols],
                            rotation=30, ha="right", fontsize=9)
        ax.set_title(
            f"Usage / Activity Metrics: Churned vs Retained — {self.dataset_name}",
            fontweight="bold")
        ax.set_ylabel("Mean Value"); ax.legend(); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        self._save(fig, idx, "usage_comparison")

    def draw_churn_by_category(self, idx, cols):
        """Churn rate bar per category value for each categorical column."""
        cols = [c for c in cols if c in self.df_raw.columns]
        if not cols:
            return
        ncols_plot = len(cols)
        fig, axes  = plt.subplots(1, ncols_plot,
                                   figsize=(6 * ncols_plot, 5))
        if ncols_plot == 1:
            axes = [axes]

        for ax, col in zip(axes, cols):
            df_c = self.df_raw[[col]].copy()
            df_c["__churn__"] = self.y.values
            rate = (df_c.groupby(col)["__churn__"]
                        .mean()
                        .sort_values(ascending=False))
            overall = self.y.mean()
            bar_colors = [C_CHURN if v > overall * 1.2 else PALETTE[2]
                           for v in rate.values]
            ax.bar(range(len(rate)), rate.values * 100,
                   color=bar_colors, edgecolor="white")
            ax.set_xticks(range(len(rate)))
            ax.set_xticklabels([str(v)[:15] for v in rate.index],
                                rotation=35, ha="right", fontsize=9)
            ax.axhline(overall * 100, color="black", linestyle="--",
                       linewidth=1.2, label=f"Overall {overall:.1%}")
            ax.set_title(f"Churn Rate by {self._label(col)}", fontweight="bold")
            ax.set_ylabel("Churn Rate (%)"); ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle(
            f"Churn Rate by Category — {self.dataset_name}",
            fontweight="bold", fontsize=13)
        plt.tight_layout()
        self._save(fig, idx, "churn_by_category")

    def draw_binary_flag_rates(self, idx, cols):
        """For binary 0/1 columns: churn rate when flag=0 vs flag=1."""
        cols = [c for c in cols if c in self.df_raw.columns][:6]
        if not cols:
            return
        rows = []
        for col in cols:
            df_c = self.df_raw[[col]].copy()
            df_c["__churn__"] = self.y.values
            df_c[col] = df_c[col].fillna(0).astype(int)
            for flag_val in [0, 1]:
                subset = df_c[df_c[col] == flag_val]
                if len(subset) > 0:
                    rows.append({
                        "column": self._label(col),
                        "flag":   f"{'Yes' if flag_val else 'No'} ({flag_val})",
                        "churn_rate": subset["__churn__"].mean() * 100,
                        "count": len(subset),
                    })
        if not rows:
            return
        df_r = pd.DataFrame(rows)

        fig, ax = plt.subplots(figsize=(max(10, len(cols) * 2), 5))
        x       = np.arange(len(cols))
        w       = 0.35
        for i, flag_val in enumerate([0, 1]):
            label_str = "Flag = No (0)" if flag_val == 0 else "Flag = Yes (1)"
            rates = []
            for col in cols:
                col_lbl = self._label(col)
                match = df_r[(df_r["column"] == col_lbl) &
                              (df_r["flag"].str.startswith("Yes" if flag_val else "No"))]
                rates.append(match["churn_rate"].values[0] if len(match) else 0)
            ax.bar(x + (i - 0.5) * w, rates, w,
                   label=label_str,
                   color=[C_RETAIN, C_CHURN][flag_val],
                   alpha=0.85, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([self._label(c) for c in cols],
                            rotation=25, ha="right", fontsize=9)
        ax.axhline(self.y.mean() * 100, color="black", linestyle="--",
                   linewidth=1.2, label=f"Overall {self.y.mean():.1%}")
        ax.set_title(
            f"Churn Rate by Binary Feature Flag — {self.dataset_name}",
            fontweight="bold")
        ax.set_ylabel("Churn Rate (%)"); ax.legend(); ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        self._save(fig, idx, "binary_flag_rates")

    def draw_temporal_trend(self, idx, metric_col):
        """Line chart of metric_col over time, total + churn split."""
        metric_col = metric_col if metric_col in self.df_raw.columns else None
        if metric_col is None:
            return
        date_col, date_series = next(iter(self.p["date_series"].items()))
        df_t = pd.DataFrame({
            date_col:    date_series,
            metric_col:  self.df_raw[metric_col].values,
            "__churn__": self.y.values,
        })
        df_t["__month__"] = date_series.dt.to_period("M")
        monthly_all    = df_t.groupby("__month__")[metric_col].sum()
        monthly_churn  = df_t[df_t["__churn__"]==1].groupby("__month__")[metric_col].sum()
        monthly_retain = df_t[df_t["__churn__"]==0].groupby("__month__")[metric_col].sum()

        all_idx = monthly_all.index.to_timestamp()
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.fill_between(all_idx, monthly_all.values, alpha=0.12, color=C_NEUTRAL)
        ax.plot(all_idx, monthly_all.values,
                color=C_NEUTRAL, linewidth=2.5, label="All Customers", marker="o", markersize=4)
        if len(monthly_churn) > 0:
            ax.plot(monthly_churn.index.to_timestamp(), monthly_churn.values,
                    color=C_CHURN, linewidth=1.8, linestyle="--",
                    label="Churned", marker="s", markersize=4)
        if len(monthly_retain) > 0:
            ax.plot(monthly_retain.index.to_timestamp(), monthly_retain.values,
                    color=C_RETAIN, linewidth=1.8, linestyle="--",
                    label="Retained", marker="^", markersize=4)

        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.xticks(rotation=45)
        ax.set_title(
            f"Monthly {self._label(metric_col)} Trend — {self.dataset_name}",
            fontweight="bold")
        ax.set_xlabel("Month"); ax.set_ylabel(self._label(metric_col))
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        self._save(fig, idx, f"temporal_trend_{metric_col[:20]}")

    # ── dispatcher ────────────────────────────────────────────────────────────

    DISPATCH = {
        "churn_overview":             "draw_churn_overview",
        "risk_segments":              "draw_risk_segments",
        "risk_distribution":          "draw_risk_distribution",
        "feature_importance":         "draw_feature_importance",
        "model_comparison":           "draw_model_comparison",
        "top_feature_distributions":  "draw_top_feature_distributions",
        "mean_comparison":            "draw_mean_comparison",
        "correlation_heatmap":        "draw_correlation_heatmap",
        "period_trend":               "draw_period_trend",
        "revenue_comparison":         "draw_revenue_comparison",
        "usage_comparison":           "draw_usage_comparison",
        "churn_by_category":          "draw_churn_by_category",
        "binary_flag_rates":          "draw_binary_flag_rates",
        "temporal_trend":             "draw_temporal_trend",
    }

    def execute_plan(self, plan):
        """Run every chart in the plan; skip silently on errors."""
        print(f"\n[Charts] Generating {len(plan)} charts for '{self.dataset_name}' ...")
        # Clear old charts from output folder
        for f in os.listdir(OUTPUT_DIR):
            if f.endswith(".png"):
                try:
                    os.remove(os.path.join(OUTPUT_DIR, f))
                except Exception:
                    pass

        for idx, chart_type, params in plan:
            method_name = self.DISPATCH.get(chart_type)
            if not method_name:
                continue
            method = getattr(self, method_name, None)
            if not method:
                continue
            try:
                method(idx, **params)
            except Exception as e:
                print(f"  [{idx:02d}] {chart_type} skipped — {e}")

        print(f"[Charts] {len(self.generated)} charts saved → {OUTPUT_DIR}")
        return self.generated


# ══════════════════════════════════════════════════════════════════════════════
# 4.  PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_all_visualizations(df: pd.DataFrame,
                            y: pd.Series,
                            df_raw: pd.DataFrame,
                            schema: dict,
                            churn_probs: np.ndarray,
                            importance_df: pd.DataFrame,
                            results: list,
                            dataset_name: str = "Dataset") -> list:
    """
    Fully automatic chart generation.
    Analyses the dataset, plans which charts are meaningful,
    then draws and saves only those charts.
    Returns list of saved file paths.
    """
    profiler = DataProfiler(df_raw=df_raw, df_feat=df,
                             y=y, schema=schema)
    planner  = ChartPlanner(profiler)
    plan     = planner.plan()
    engine   = ChartEngine(profiler=profiler,
                            churn_probs=churn_probs,
                            importance_df=importance_df,
                            results=results,
                            dataset_name=dataset_name)
    return engine.execute_plan(plan)