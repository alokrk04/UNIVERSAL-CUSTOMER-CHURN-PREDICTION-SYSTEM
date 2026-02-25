"""
╔══════════════════════════════════════════════════════════════════╗
║      UNIVERSAL CUSTOMER CHURN PREDICTION SYSTEM                  ║
║      Works with ANY CSV dataset — auto-detects schema            ║
╠══════════════════════════════════════════════════════════════════╣
║  Usage:                                                          ║
║    python main.py --data path/to/your_file.csv                   ║
║    python main.py --data data/telecom_churn_data.csv             ║
║    python main.py --data data/bank_churn.csv                     ║
║    python main.py --data data/ecommerce_customers.csv            ║
║                                                                  ║
║  Optional flags:                                                 ║
║    --name "My Dataset"    Label for charts                       ║
║    --no-charts            Skip visualization (faster)            ║
║    --skip-xgb             Skip XGBoost                           ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["OMP_NUM_THREADS"] = "1"

import sys, argparse, warnings, time
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config                 import OUTPUT_DIR, MODELS_DIR, DATA_DIR
from auto_detector          import SchemaDetector
from universal_preprocessor import run_universal_etl, split_dataset
from universal_features     import engineer_features, select_features
from churn_models           import (RandomForestChurnModel,
                                    EnsembleChurnModel)
from predictor              import ChurnPredictor
from universal_visualizer   import run_all_visualizations

try:
    from churn_models import XGBoostChurnModel
    HAS_XGB = True
except Exception:
    HAS_XGB = False


# ══════════════════════════════════════════════════════════════════
def banner(text, width=62):
    print("\n" + "━"*width)
    print(f"  {text}")
    print("━"*width)


def parse_args():
    p = argparse.ArgumentParser(
        description="Universal Churn Prediction System")
    p.add_argument("--data",     type=str, default=None,
                   help="Path to your CSV file")
    p.add_argument("--name",     type=str, default=None,
                   help="Dataset display name")
    p.add_argument("--no-charts",action="store_true",
                   help="Skip chart generation")
    p.add_argument("--skip-xgb", action="store_true",
                   help="Skip XGBoost training")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════
def find_data_file(args_data):
    """Resolve the CSV path — from arg, or auto-discover in data/"""
    if args_data:
        if os.path.exists(args_data):
            return args_data
        # Try relative to data/ folder
        candidate = os.path.join(DATA_DIR, args_data)
        if os.path.exists(candidate):
            return candidate
        raise FileNotFoundError(f"File not found: {args_data}")

    # Auto-discover: find any CSV in data/
    csvs = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    if csvs:
        path = os.path.join(DATA_DIR, csvs[0])
        print(f"[Auto] No --data flag given. Using: {path}")
        return path

    raise FileNotFoundError(
        "No CSV file found. Place your CSV in the 'data/' folder "
        "or run: python main.py --data path/to/file.csv"
    )


# ══════════════════════════════════════════════════════════════════
def main():
    args         = parse_args()
    t_start      = time.time()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR,   exist_ok=True)

    # ── Resolve data file ─────────────────────────────────────────
    csv_path     = find_data_file(args.data)
    dataset_name = args.name or os.path.splitext(
                    os.path.basename(csv_path))[0].replace("_"," ").title()

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║        UNIVERSAL CHURN PREDICTION SYSTEM                         ║         
║  Dataset : {dataset_name:<54}║            
║  File    : {os.path.basename(csv_path):<54}║          
╚══════════════════════════════════════════════════════════════════╝""")

    # ── Step 1: Auto-detect + ETL ─────────────────────────────────
    banner("Step 1: Auto-detecting Schema & Running ETL")
    df_clean, y, df_raw, schema = run_universal_etl(csv_path)

    # ── Step 2: Feature Engineering ───────────────────────────────
    banner("Step 2: Feature Engineering")
    df_feat = engineer_features(df_clean, schema, df_raw)
    df_feat = select_features(df_feat, y, max_features=150)
    feature_cols = df_feat.columns.tolist()
    print(f"[Features] Final feature matrix: {df_feat.shape}")

    # ── Step 3: Train / Val / Test Split ──────────────────────────
    banner("Step 3: Dataset Split")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        df_feat, y)

    # ── Step 4: Model Training ─────────────────────────────────────
    results = []

    ## 4a. Random Forest
    banner("Step 4a: Random Forest")
    rf = RandomForestChurnModel()
    rf.fit(X_train, y_train, X_val, y_val)
    rf_metrics = rf.evaluate(X_test, y_test)
    rf.save()
    results.append(rf_metrics)

    fi = rf.feature_importance(top_n=20)
    print(f"\nTop 10 features:\n"
          f"{fi.head(10).to_string(index=False)}")

    ## 4b. XGBoost
    xgb = None
    if HAS_XGB and not args.skip_xgb:
        banner("Step 4b: XGBoost")
        try:
            xgb = XGBoostChurnModel()
            xgb.fit(X_train, y_train, X_val, y_val)
            xgb_metrics = xgb.evaluate(X_test, y_test)
            xgb.save()
            results.append(xgb_metrics)
        except Exception as e:
            print(f"[SKIP] XGBoost failed: {e}"); xgb = None
    elif not HAS_XGB:
        print("[INFO] XGBoost not installed. "
              "Run: pip install xgboost  for better accuracy.")

    ## 4c. Ensemble
    banner("Step 4c: Ensemble (Soft Voting)")
    weights = (0.35, 0.65) if xgb else (1.0, 0.0)
    ensemble = EnsembleChurnModel(rf=rf, xgb=xgb, weights=weights)
    ens_metrics = ensemble.evaluate(X_test, y_test)
    results.append(ens_metrics)

    # ── Step 5: Visualisations ─────────────────────────────────────
    if not args.no_charts:
        banner("Step 5: Generating Charts")
        best_model   = ensemble if xgb else rf
        churn_probs  = best_model.predict_proba(X_test)
        run_all_visualizations(
            df=df_feat, y=y, df_raw=df_raw, schema=schema,
            churn_probs=churn_probs, importance_df=fi,
            results=results, dataset_name=dataset_name
        )

    # ── Step 6: Export Predictions ─────────────────────────────────
    banner("Step 6: Exporting Predictions")
    best_model  = ensemble if xgb else rf
    predictor   = ChurnPredictor.from_model(best_model, feature_cols)

    # Score ALL customers (not just test set)
    all_preds = predictor.predict_batch(df_feat)
    all_preds["churn_probability"] = best_model.predict_proba(df_feat)
    all_preds["churn_prediction"]  = (all_preds["churn_probability"] >= 0.5).astype(int)

    # Add original ID column back if found
    id_col = schema.get("id_col")
    if id_col and id_col in df_raw.columns:
        all_preds.insert(0, id_col, df_raw[id_col].values)

    # Save full prediction CSV
    pred_path = os.path.join(OUTPUT_DIR, "churn_predictions_full.csv")
    all_preds[
        ([id_col] if id_col and id_col in all_preds.columns else []) +
        ["churn_probability","churn_prediction","risk_level","advice"]
    ].to_csv(pred_path, index=False)
    print(f"[Export] Full predictions → {pred_path}")

    # Save only high-risk customers
    high_risk_df = all_preds[all_preds["churn_prediction"] == 1].copy()
    high_risk_path = os.path.join(OUTPUT_DIR, "high_risk_customers.csv")
    high_risk_df[
        ([id_col] if id_col and id_col in high_risk_df.columns else []) +
        ["churn_probability","risk_level","advice"]
    ].sort_values("churn_probability", ascending=False).to_csv(
        high_risk_path, index=False)
    print(f"[Export] High-risk customers → {high_risk_path}")

    # ── Step 7: Sample Predictions Demo ───────────────────────────
    banner("Step 7: Sample Real-Time Predictions")
    print("\nFirst 5 test customers:")
    sample = X_test.head(5)
    for i, (idx, row) in enumerate(sample.iterrows()):
        r = predictor.predict_single(row.to_dict())
        id_str = f"ID {df_raw.iloc[idx][id_col]}" if (id_col and id_col in df_raw.columns) else f"Customer #{i+1}"
        print(f"  {id_str:25s} | Prob: {r['churn_probability']:.1%} | "
              f"{r['risk_level']}")

    # ── Final Summary ─────────────────────────────────────────────
    elapsed = time.time() - t_start
    banner("PIPELINE COMPLETE ✓")

    res_df = pd.DataFrame(results)
    print(f"\n{'Model':<15} {'Accuracy':>10} {'ROC-AUC':>10} "
          f"{'F1':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 65)
    for _, row in res_df.iterrows():
        print(f"  {row['model']:<13} {row['accuracy']:>10.4f} "
              f"{row['roc_auc']:>10.4f} {row['f1']:>10.4f} "
              f"{row['precision']:>10.4f} {row['recall']:>10.4f}")

    print(f"""
  Dataset        : {dataset_name}
  Total Customers: {len(df_feat):,}
  Churned        : {int(y.sum()):,}  ({y.mean():.1%})
  Best AUC       : {max(r['roc_auc'] for r in results):.4f}
  Time taken     : {elapsed:.1f}s

  Outputs saved to: {OUTPUT_DIR}/
    ├── churn_predictions_full.csv  (all {len(df_feat):,} customers scored)
    ├── high_risk_customers.csv     ({int(all_preds['churn_prediction'].sum()):,} at-risk customers)
    └── 01–10 chart PNG files
""")


# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
