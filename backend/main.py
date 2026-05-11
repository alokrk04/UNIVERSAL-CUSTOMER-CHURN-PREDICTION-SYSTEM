"""
╔══════════════════════════════════════════════════════════════════╗
║      UNIVERSAL CHURN PREDICTION SYSTEM - FastAPI Backend         ║
║      RESTful API for the web application                         ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import json
import threading
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import warnings

warnings.filterwarnings("ignore")

# ── Add parent directory to path (for imports from main churn system) ──
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import OUTPUT_DIR, MODELS_DIR, DATA_DIR
from auto_detector import SchemaDetector
from universal_preprocessor import run_universal_etl, split_dataset
from universal_features import engineer_features, select_features
from churn_models import RandomForestChurnModel, EnsembleChurnModel
from predictor import ChurnPredictor

try:
    from churn_models import XGBoostChurnModel
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# ══════════════════════════════════════════════════════════════════
# FastAPI Setup
# ══════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Churn Prediction System API",
    description="Universal Customer Churn Prediction System",
    version="1.0.0"
)

# ── CORS Configuration ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Create necessary directories ──
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "backend", "uploads"), exist_ok=True)

UPLOAD_DIR = os.path.join(BASE_DIR, "backend", "uploads")

# ── Global state for tracking pipeline execution ──
PIPELINE_STATE = {
    "status": "idle",  # idle, uploading, running, complete, error
    "progress": 0,
    "message": "",
    "dataset_name": "",
    "csv_path": "",
    "results": None,
    "schema": None,
    "error": None,
    "start_time": None,
    "end_time": None,
    "duration": 0,
}

STATE_LOCK = threading.Lock()

# ══════════════════════════════════════════════════════════════════
# Utility Functions
# ══════════════════════════════════════════════════════════════════

def update_state(status: str = None, progress: int = None, 
                 message: str = None, **kwargs):
    """Thread-safe state update."""
    with STATE_LOCK:
        if status is not None:
            PIPELINE_STATE["status"] = status
        if progress is not None:
            PIPELINE_STATE["progress"] = progress
        if message is not None:
            PIPELINE_STATE["message"] = message
        for key, value in kwargs.items():
            PIPELINE_STATE[key] = value


def run_pipeline_background(csv_path: str, dataset_name: str, 
                            skip_xgb: bool = False):
    """Execute the full churn prediction pipeline in background."""
    try:
        update_state(status="running", progress=0, message="Initializing pipeline...")
        start_time = time.time()

        # ── Step 1: Auto-detect + ETL ──
        update_state(progress=15, message="Auto-detecting schema...")
        df_clean, y, df_raw, schema = run_universal_etl(csv_path)

        # ── Step 2: Feature Engineering ──
        update_state(progress=30, message="Engineering features...")
        df_feat = engineer_features(df_clean, schema, df_raw)
        df_feat = select_features(df_feat, y, max_features=150)
        feature_cols = df_feat.columns.tolist()

        # ── Step 3: Train / Val / Test Split ──
        update_state(progress=45, message="Splitting dataset...")
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
            df_feat, y)

        # ── Step 4: Model Training ──
        results = []

        ## 4a. Random Forest
        update_state(progress=50, message="Training Random Forest...")
        rf = RandomForestChurnModel()
        rf.fit(X_train, y_train, X_val, y_val)
        rf_metrics = rf.evaluate(X_test, y_test)
        rf.save()
        results.append(rf_metrics)
        fi = rf.feature_importance(top_n=20)

        ## 4b. XGBoost
        xgb = None
        if HAS_XGB and not skip_xgb:
            update_state(progress=65, message="Training XGBoost...")
            try:
                xgb = XGBoostChurnModel()
                xgb.fit(X_train, y_train, X_val, y_val)
                xgb_metrics = xgb.evaluate(X_test, y_test)
                xgb.save()
                results.append(xgb_metrics)
            except Exception as e:
                print(f"[XGBoost] Failed: {e}")
                xgb = None

        ## 4c. Ensemble
        update_state(progress=75, message="Building Ensemble...")
        weights = (0.35, 0.65) if xgb else (1.0, 0.0)
        ensemble = EnsembleChurnModel(rf=rf, xgb=xgb, weights=weights)
        ens_metrics = ensemble.evaluate(X_test, y_test)
        results.append(ens_metrics)

        # ── Step 5: Generate Visualizations ──
        update_state(progress=80, message="Generating visualizations...")
        from universal_visualizer import run_all_visualizations
        best_model = ensemble if xgb else rf
        churn_probs = best_model.predict_proba(X_test)
        run_all_visualizations(
            df=df_feat, y=y, df_raw=df_raw, schema=schema,
            churn_probs=churn_probs, importance_df=fi,
            results=results, dataset_name=dataset_name
        )

        # ── Step 6: Export Predictions ──
        update_state(progress=85, message="Exporting predictions...")
        best_model = ensemble if xgb else rf
        predictor = ChurnPredictor.from_model(best_model, feature_cols)

        # Score ALL customers
        all_preds = predictor.predict_batch(df_feat)
        all_preds["churn_probability"] = best_model.predict_proba(df_feat)
        all_preds["churn_prediction"] = (
            all_preds["churn_probability"] >= 0.5).astype(int)

        # Add ID column if found
        id_col = schema.get("id_col")
        if id_col and id_col in df_raw.columns:
            all_preds.insert(0, id_col, df_raw[id_col].values)

        # Save full predictions
        pred_path = os.path.join(OUTPUT_DIR, "churn_predictions_full.csv")
        all_preds[
            ([id_col] if id_col and id_col in all_preds.columns else []) +
            ["churn_probability", "churn_prediction", "risk_level", "advice"]
        ].to_csv(pred_path, index=False)

        # Save high-risk customers
        high_risk_df = all_preds[all_preds["churn_prediction"] == 1].copy()
        high_risk_path = os.path.join(OUTPUT_DIR, "high_risk_customers.csv")
        high_risk_df[
            ([id_col] if id_col and id_col in high_risk_df.columns else []) +
            ["churn_probability", "risk_level", "advice"]
        ].sort_values("churn_probability", ascending=False).to_csv(
            high_risk_path, index=False)

        # ── Prepare results ──
        update_state(progress=90, message="Finalizing results...")

        best_auc = max(r["roc_auc"] for r in results)
        churn_count = int(y.sum())
        churn_rate = y.mean()
        total_customers = len(df_feat)
        at_risk_count = int(all_preds["churn_prediction"].sum())

        summary = {
            "dataset_name": dataset_name,
            "total_customers": total_customers,
            "churned_customers": churn_count,
            "churn_rate": float(churn_rate),
            "at_risk_customers": at_risk_count,
            "best_model": "Ensemble" if xgb else "Random Forest",
            "best_auc": float(best_auc),
            "models_trained": ["Random Forest"] + 
                              (["XGBoost"] if xgb else []) + 
                              ["Ensemble"],
            "schema": {
                "id_col": schema.get("id_col"),
                "churn_col": schema.get("churn_col"),
                "date_cols": schema.get("date_cols"),
                "numeric_cols": schema.get("numeric_cols", [])[:10],
                "categorical_cols": schema.get("categorical_cols", [])[:10],
                "n_rows": schema.get("n_rows"),
                "n_cols": schema.get("n_cols"),
            },
            "model_results": results,
        }

        end_time = time.time()
        duration = end_time - start_time

        update_state(
            status="complete",
            progress=100,
            message="Pipeline complete!",
            results=summary,
            schema=summary["schema"],
            duration=duration,
            end_time=end_time,
        )

    except Exception as e:
        error_msg = str(e)
        print(f"[Pipeline Error] {error_msg}")
        import traceback
        traceback.print_exc()
        update_state(
            status="error",
            message="Pipeline failed",
            error=error_msg
        )


# ══════════════════════════════════════════════════════════════════
# API Endpoints
# ══════════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "xgboost_available": HAS_XGB,
    }


@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Upload a CSV file and prepare for pipeline execution."""
    try:
        update_state(status="uploading", progress=5, message="Uploading file...")

        if not file.filename.endswith(".csv"):
            raise HTTPException(
                status_code=400,
                detail="Only CSV files are supported"
            )

        # Save uploaded file
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        # Validate CSV
        try:
            df = pd.read_csv(file_path)
            if len(df) == 0:
                raise ValueError("CSV is empty")
        except Exception as e:
            os.remove(file_path)
            raise HTTPException(
                status_code=400,
                detail=f"Invalid CSV: {str(e)}"
            )

        dataset_name = file.filename.replace(".csv", "").replace("_", " ").title()

        update_state(
            status="idle",
            progress=10,
            message="File uploaded successfully",
            csv_path=file_path,
            dataset_name=dataset_name,
        )

        return {
            "status": "success",
            "filename": file.filename,
            "dataset_name": dataset_name,
            "rows": len(df),
            "columns": len(df.columns),
        }

    except HTTPException:
        raise
    except Exception as e:
        update_state(status="error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/run-pipeline")
async def run_pipeline(skip_xgb: bool = False, background_tasks: BackgroundTasks = None):
    """Start the churn prediction pipeline (runs in background)."""
    try:
        if PIPELINE_STATE["csv_path"] == "":
            raise HTTPException(
                status_code=400,
                detail="No CSV file uploaded. Please upload a file first."
            )

        if PIPELINE_STATE["status"] == "running":
            raise HTTPException(
                status_code=400,
                detail="Pipeline is already running"
            )

        # Reset state
        update_state(
            status="running",
            progress=0,
            message="Starting pipeline...",
            error=None,
            start_time=time.time(),
        )

        # Run pipeline in background
        if background_tasks:
            background_tasks.add_task(
                run_pipeline_background,
                PIPELINE_STATE["csv_path"],
                PIPELINE_STATE["dataset_name"],
                skip_xgb
            )
        else:
            # Fallback: run in thread
            thread = threading.Thread(
                target=run_pipeline_background,
                args=(
                    PIPELINE_STATE["csv_path"],
                    PIPELINE_STATE["dataset_name"],
                    skip_xgb
                )
            )
            thread.daemon = True
            thread.start()

        return {
            "status": "pipeline_started",
            "message": "Pipeline is running in background"
        }

    except HTTPException:
        raise
    except Exception as e:
        update_state(status="error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status")
async def get_status():
    """Get current pipeline execution status."""
    return {
        "status": PIPELINE_STATE["status"],
        "progress": PIPELINE_STATE["progress"],
        "message": PIPELINE_STATE["message"],
        "dataset_name": PIPELINE_STATE["dataset_name"],
        "results": PIPELINE_STATE["results"],
        "schema": PIPELINE_STATE["schema"],
        "error": PIPELINE_STATE["error"],
        "duration": PIPELINE_STATE["duration"],
    }


@app.get("/api/results")
async def get_results():
    """Fetch detailed results after pipeline completion."""
    if PIPELINE_STATE["status"] != "complete":
        raise HTTPException(
            status_code=400,
            detail="Pipeline has not completed yet"
        )

    return {
        "results": PIPELINE_STATE["results"],
        "duration": PIPELINE_STATE["duration"],
    }


@app.get("/api/visualizations")
async def list_visualizations():
    """List all generated visualization images."""
    try:
        images = []
        if os.path.exists(OUTPUT_DIR):
            for file in sorted(os.listdir(OUTPUT_DIR)):
                if file.endswith(".png"):
                    images.append({
                        "filename": file,
                        "url": f"/api/image/{file}",
                    })
        return {"images": images}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/image/{filename}")
async def get_image(filename: str):
    """Serve a visualization image."""
    try:
        file_path = os.path.join(OUTPUT_DIR, filename)
        if not os.path.exists(file_path) or not filename.endswith(".png"):
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(file_path, media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions/all")
async def get_all_predictions():
    """Fetch all predictions as JSON."""
    try:
        pred_path = os.path.join(OUTPUT_DIR, "churn_predictions_full.csv")
        if not os.path.exists(pred_path):
            raise HTTPException(
                status_code=404,
                detail="Predictions file not found. Run pipeline first."
            )

        df = pd.read_csv(pred_path)
        
        # Clean data for JSON serialization
        df = df.where(pd.notna(df), None)
        
        return {
            "total_records": len(df),
            "predictions": df.to_dict(orient="records"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions/high-risk")
async def get_high_risk_predictions():
    """Fetch high-risk predictions."""
    try:
        high_risk_path = os.path.join(OUTPUT_DIR, "high_risk_customers.csv")
        if not os.path.exists(high_risk_path):
            raise HTTPException(
                status_code=404,
                detail="High-risk file not found. Run pipeline first."
            )

        df = pd.read_csv(high_risk_path)
        df = df.where(pd.notna(df), None)

        return {
            "total_high_risk": len(df),
            "predictions": df.to_dict(orient="records"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions/search")
async def search_predictions(query: str = Query(...)):
    """Search predictions by customer ID or other fields."""
    try:
        pred_path = os.path.join(OUTPUT_DIR, "churn_predictions_full.csv")
        if not os.path.exists(pred_path):
            raise HTTPException(status_code=404, detail="Predictions not found")

        df = pd.read_csv(pred_path)
        
        # Search across all columns
        mask = df.astype(str).apply(
            lambda x: x.str.contains(query, case=False, regex=False)
        ).any(axis=1)
        
        results = df[mask]
        results = results.where(pd.notna(results), None)

        return {
            "total_results": len(results),
            "predictions": results.to_dict(orient="records"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions/download")
async def download_predictions(file_type: str = "all"):
    """Download predictions as CSV."""
    try:
        if file_type == "high_risk":
            file_path = os.path.join(OUTPUT_DIR, "high_risk_customers.csv")
        else:
            file_path = os.path.join(OUTPUT_DIR, "churn_predictions_full.csv")

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(
            file_path,
            media_type="text/csv",
            filename=os.path.basename(file_path)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reset")
async def reset_pipeline():
    """Reset the pipeline state and clear uploads."""
    try:
        with STATE_LOCK:
            PIPELINE_STATE.update({
                "status": "idle",
                "progress": 0,
                "message": "",
                "dataset_name": "",
                "csv_path": "",
                "results": None,
                "schema": None,
                "error": None,
                "start_time": None,
                "end_time": None,
                "duration": 0,
            })

        # Clean up uploaded files
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
            os.makedirs(UPLOAD_DIR, exist_ok=True)

        return {"status": "reset_complete"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════
# Main Entry Point
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║        CHURN PREDICTION SYSTEM - FastAPI Backend                 ║
║        Starting server on http://localhost:8000                   ║
║        API docs available at http://localhost:8000/docs          ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

