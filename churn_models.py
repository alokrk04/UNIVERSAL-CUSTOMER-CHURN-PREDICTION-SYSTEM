"""
churn_models.py - Churn Prediction Models
==========================================
Random Forest, XGBoost (if available), and Ensemble.
Works on any feature matrix.
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                              recall_score, accuracy_score,
                              classification_report)
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RF_PARAMS, MODELS_DIR

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def evaluate_model(name, y_true, y_pred, y_prob):
    try:
        auc = round(roc_auc_score(y_true, y_prob), 4)
    except ValueError:
        auc = 0.5   # only one class in test set

    metrics = {
        "model":     name,
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "roc_auc":   auc,
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
    }
    print(f"\n{'='*55}")
    print(f"  {name} — Test Results")
    print(f"{'='*55}")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k:<12}: {v:.4f}")
    try:
        print(f"\n{classification_report(y_true, y_pred, target_names=['Retained','Churned'])}")
    except Exception:
        print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    return metrics


class RandomForestChurnModel:
    def __init__(self):
        self.model = RandomForestClassifier(**RF_PARAMS)
        self.feature_names = None
        self.name = "RandomForest"

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.feature_names = (X_train.columns.tolist()
                               if hasattr(X_train, 'columns') else None)
        print(f"[{self.name}] Training on {len(X_train):,} samples ...")
        self.model.fit(X_train, y_train)
        if X_val is not None:
            try:
                proba = self.model.predict_proba(X_val)
                if proba.shape[1] > 1:
                    val_auc = roc_auc_score(y_val, proba[:,1])
                    val_f1  = f1_score(y_val, self.model.predict(X_val), zero_division=0)
                    print(f"[{self.name}] Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f}")
                else:
                    print(f"[{self.name}] Only one class in training — check churn derivation")
            except Exception as e:
                print(f"[{self.name}] Val eval skipped: {e}")
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:,1]

    def evaluate(self, X_test, y_test):
        return evaluate_model(self.name, y_test,
                               self.predict(X_test), self.predict_proba(X_test))

    def feature_importance(self, top_n=20):
        if not self.feature_names:
            raise ValueError("Not yet trained.")
        return (pd.DataFrame({
                    "feature":    self.feature_names,
                    "importance": self.model.feature_importances_
                })
                .sort_values("importance", ascending=False)
                .head(top_n)
                .reset_index(drop=True))

    def save(self, path=None):
        path = path or os.path.join(MODELS_DIR, "rf_model.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[{self.name}] Saved → {path}")

    @classmethod
    def load(cls, path=None):
        path = path or os.path.join(MODELS_DIR, "rf_model.pkl")
        with open(path, "rb") as f:
            return pickle.load(f)


class XGBoostChurnModel:
    def __init__(self):
        if not HAS_XGB:
            raise ImportError("pip install xgboost")
        from config import XGB_PARAMS
        self.model = XGBClassifier(**XGB_PARAMS)
        self.name  = "XGBoost"

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        print(f"[{self.name}] Training on {len(X_train):,} samples ...")
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=50)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:,1]

    def evaluate(self, X_test, y_test):
        return evaluate_model(self.name, y_test,
                               self.predict(X_test), self.predict_proba(X_test))

    def save(self, path=None):
        path = path or os.path.join(MODELS_DIR, "xgb_model.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[{self.name}] Saved → {path}")

    @classmethod
    def load(cls, path=None):
        path = path or os.path.join(MODELS_DIR, "xgb_model.pkl")
        with open(path, "rb") as f:
            return pickle.load(f)


class EnsembleChurnModel:
    def __init__(self, rf=None, xgb=None, weights=(0.40, 0.60)):
        self.models  = {"RF": rf, "XGB": xgb}
        self.weights = weights
        self.name    = "Ensemble"

    def predict_proba(self, X):
        probs, wts = [], []
        for (n, m), w in zip(self.models.items(), self.weights):
            if m is not None:
                probs.append(m.predict_proba(X) * w)
                wts.append(w)
        return np.sum(probs, axis=0) / sum(wts)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def evaluate(self, X_test, y_test):
        return evaluate_model(self.name, y_test,
                               self.predict(X_test), self.predict_proba(X_test))