"""
config.py - Universal Churn Prediction System
==============================================
Central settings. The system auto-detects your dataset schema,
so you rarely need to change anything here.
"""
import os

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ── Model Settings ─────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.20
VAL_SIZE     = 0.10

RF_PARAMS = {
    "n_estimators":     300,
    "max_depth":        None,
    "min_samples_split":5,
    "min_samples_leaf": 2,
    "class_weight":     "balanced",
    "random_state":     RANDOM_STATE,
    "n_jobs":           1,            # 1 = safe on macOS; -1 = faster on Linux
}

XGB_PARAMS = {
    "n_estimators":   500,
    "learning_rate":  0.05,
    "max_depth":      6,
    "subsample":      0.8,
    "colsample_bytree": 0.8,
    "eval_metric":    "logloss",
    "random_state":   RANDOM_STATE,
}

CHURN_RISK_THRESHOLD = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR,   exist_ok=True)
