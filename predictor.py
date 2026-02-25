"""
predictor.py - Real-Time Churn Scorer & Report Generator
=========================================================
Scores any customer record or DataFrame and outputs:
  - Churn probability
  - Risk band (Low / Medium / High / Critical)
  - Full results CSV exported to outputs/
"""

import numpy as np
import pandas as pd
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CHURN_RISK_THRESHOLD, OUTPUT_DIR

RISK_BANDS = [
    (0.00, 0.30, "🟢 Low",      "Customer is active and unlikely to churn."),
    (0.30, 0.60, "🟡 Medium",   "Some risk signals — consider a retention offer."),
    (0.60, 0.80, "🟠 High",     "Strong churn signals — proactive outreach needed."),
    (0.80, 1.01, "🔴 Critical", "Imminent churn — immediate intervention required."),
]


def risk_band(prob):
    for lo, hi, label, advice in RISK_BANDS:
        if lo <= prob < hi:
            return label, advice
    return "🔴 Critical", "Immediate intervention required."


class ChurnPredictor:
    def __init__(self, model, feature_cols, threshold=CHURN_RISK_THRESHOLD):
        self.model        = model
        self.feature_cols = feature_cols
        self.threshold    = threshold

    def _prep(self, record):
        if isinstance(record, dict):
            row = pd.DataFrame([record])
        elif isinstance(record, pd.Series):
            row = record.to_frame().T
        else:
            row = record.copy()
        for col in self.feature_cols:
            if col not in row.columns:
                row[col] = 0
        return row[self.feature_cols].fillna(0).values.astype(float)

    def predict_single(self, record):
        arr  = self._prep(record)
        prob = float(self.model.predict_proba(arr)[0]
                     if hasattr(self.model.predict_proba(arr), '__len__')
                     else self.model.predict_proba(arr))
        label, advice = risk_band(prob)
        return {
            "churn_probability": round(prob, 4),
            "churn_prediction":  int(prob >= self.threshold),
            "risk_level":        label,
            "advice":            advice,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        arr   = self._prep(df)
        probs = self.model.predict_proba(arr)
        out   = df.copy().reset_index(drop=True)
        out["churn_probability"] = np.round(probs, 4)
        out["churn_prediction"]  = (probs >= self.threshold).astype(int)
        out["risk_level"]        = [risk_band(p)[0] for p in probs]
        out["advice"]            = [risk_band(p)[1] for p in probs]
        return out

    def export_results(self, df: pd.DataFrame, id_col=None,
                        filename="churn_predictions.csv"):
        """Score all rows and save a predictions CSV."""
        results = self.predict_batch(df)
        # Bring ID column to front if available
        cols = results.columns.tolist()
        if id_col and id_col in cols:
            cols = [id_col] + [c for c in cols if c != id_col]
            results = results[cols]
        path = os.path.join(OUTPUT_DIR, filename)
        results.to_csv(path, index=False)
        print(f"[Predictor] Predictions saved → {path}")
        return results

    def score_report(self, record):
        r = self.predict_single(record)
        lines = [
            "="*55,
            "  CUSTOMER CHURN RISK REPORT",
            "="*55,
            f"  Churn Probability : {r['churn_probability']:.1%}",
            f"  Risk Level        : {r['risk_level']}",
            f"  Prediction        : {'⚠  CHURN' if r['churn_prediction'] else '✓  RETAIN'}",
            f"  Advice            : {r['advice']}",
            "="*55,
        ]
        return "\n".join(lines)

    @classmethod
    def from_model(cls, model, feature_cols):
        return cls(model, feature_cols)
