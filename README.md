# UNIVERSAL-CUSTOMER-CHURN-PREDICTION-SYSTEM
A Universal Customer Churn Prediction System is an advanced analytics framework designed to identify which customers are likely to stop using a service or product. The "universal" system is built to be industry-agnostic, capable of processing diverse data streams—from banking transactions to telecom usage—using a unified machine learning pipeline.


# 🔄 Universal Churn Predictor

> **Drop in any customer dataset. Get churn predictions, risk scores, and data-driven charts — automatically.**

A production-ready, end-to-end machine learning pipeline that works with **any CSV dataset** regardless of domain or column structure. No configuration required. The system auto-detects your schema, derives churn labels if none exist, engineers features, trains models, and generates a fully customised set of visualisations — all from a single command.

---

## ✨ Features

### 🧠 Fully Automatic Schema Detection
- Detects **customer ID**, **churn label**, **date**, **revenue**, and **usage** columns automatically using keyword matching and statistical heuristics
- Handles churn labels in any format: `1/0`, `True/False`, `Yes/No`, `Churn/No Churn`, `Exited/Retained`, and more

### 🏷️ Smart Churn Label Derivation
If your dataset has no churn column, the system derives one using a cascading strategy:
1. **Recency-based** — customers who haven't engaged recently are marked as churned
2. **Inactivity-based** — customers with zero or near-zero usage metrics
3. **Low-value-based** — customers in the bottom revenue percentile
4. **Activity score** — bottom 15–20% of overall activity
5. **Auto-recovery** — if the derived churn rate is unrealistic (< 1% or > 90%), it automatically tries the next strategy

### ⚙️ Adaptive Feature Engineering
Automatically engineers domain-agnostic features on any dataset:
- **Row-level statistics** — mean, std, min, max, range, zero-count
- **Revenue trend** — decline ratio across time periods
- **Usage trend** — activity change detection
- **Recharge / engagement frequency**
- **Data usage score** (composite index)
- **Temporal features** — days since last activity, cyclical month encoding
- **Ratio features** — revenue-per-usage and cross-column interactions
- **Inactivity score** — normalised 0–1 disengagement index
- **Feature selection** — correlation-based pruning to top 150 features

### 🤖 ML Models
| Model | Description |
|---|---|
| **Random Forest** | Ensemble of decision trees with class-balancing; always available |
| **XGBoost** | Gradient boosted trees; optional, install with `pip install xgboost` |
| **Ensemble** | Soft-voting combination of RF + XGB with configurable weights |

### 📊 Data-Driven Chart Engine (Zero Hardcoding)
The visualiser has three components that work together:

- **`DataProfiler`** — scans the dataset and builds a rich profile of every column type, variance, period families, and relationships
- **`ChartPlanner`** — examines the profile and decides which chart types are meaningful for this specific dataset (unused chart types are silently skipped)
- **`ChartEngine`** — draws each chart using your dataset's actual column names for all titles, labels, and legends

Charts generated vary by dataset and can include:
- Churn overview (pie + count bar)
- Risk segmentation (4 bands: Low / Medium / High / Critical)
- Predicted churn probability distribution
- Feature importance (top 20 predictors)
- Model performance comparison
- Top feature distributions — KDE split by churn status
- Key metrics mean comparison
- Correlation heatmap
- Multi-period trend lines (auto-detected column families like `arpu_6/7/8`)
- Revenue & usage metric comparisons
- Churn rate by categorical column
- Binary flag analysis
- Temporal trend over time (if date column present)

### 📤 Prediction Exports
Every run produces two ready-to-use CSVs:
- **`churn_predictions_full.csv`** — every customer scored with probability + risk band
- **`high_risk_customers.csv`** — only at-risk customers, sorted by highest churn probability

Risk bands:
| Band | Probability | Action |
|---|---|---|
| 🟢 Low | 0–30% | No action needed |
| 🟡 Medium | 30–60% | Consider a retention offer |
| 🟠 High | 60–80% | Proactive outreach needed |
| 🔴 Critical | 80–100% | Immediate intervention required |

---

## 🗂️ Project Structure

```
Universal Churn Predictor/
│
├── main.py                     # Entry point — runs the full pipeline
├── config.py                   # Central settings (model params, paths)
├── auto_detector.py            # Schema detection engine
├── universal_preprocessor.py   # ETL — loading, cleaning, churn derivation, splitting
├── universal_features.py       # Adaptive feature engineering
├── universal_visualizer.py     # Data-driven chart engine
├── churn_models.py             # Random Forest, XGBoost, Ensemble models
├── predictor.py                # Real-time scorer + risk report + CSV exporter
├── requirements.txt            # Python dependencies
│
├── data/                       # ← Put your CSV files here
│   └── your_dataset.csv
│
├── models/                     # Saved trained models (auto-created)
│   ├── rf_model.pkl
│   └── xgb_model.pkl
│
└── outputs/                    # All results saved here (auto-created)
    ├── churn_predictions_full.csv
    ├── high_risk_customers.csv
    └── 01_churn_overview.png … (N charts, varies by dataset)
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/your-username/universal-churn-predictor.git
cd universal-churn-predictor
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your dataset
```bash
cp /path/to/your_data.csv data/
```

### 5. Run
```bash
python main.py --data data/your_data.csv --name "My Dataset"
```

That's it. Results will be in the `outputs/` folder.

---

## 🖥️ Usage

```
python main.py [OPTIONS]

Options:
  --data PATH        Path to your CSV file (or filename inside data/)
  --name TEXT        Display name for charts and reports
  --no-charts        Skip chart generation (faster, predictions only)
  --skip-xgb         Skip XGBoost training (use if not installed)
```

### Examples
```bash
# Telecom churn dataset
python main.py --data data/telecom_churn.csv --name "Telecom Q3"

# Bank customer attrition
python main.py --data data/bank_customers.csv --name "Bank Churn"

# E-commerce with no churn label — system will derive one
python main.py --data data/ecommerce_orders.csv --name "E-Commerce"

# Skip charts for a quick predictions run
python main.py --data data/customers.csv --no-charts

# Use any CSV anywhere on your machine
python main.py --data /Users/you/Downloads/cell2cellholdout.csv
```

---

## 📋 Supported Dataset Types

The system has been tested on — and works with:

| Domain | Dataset Type | Churn Detection |
|---|---|---|
| Telecom | Multi-month usage + recharge data | Auto-derived from inactivity |
| Banking | Customer demographics + transactions | Label column (`Exited`) |
| E-commerce | Order history + spend | Auto-derived from recency |
| SaaS / Subscription | Plan usage + login activity | Label column (`Churned`) |
| Retail | Purchase frequency + RFM | Auto-derived from low value |
| Insurance | Policy + claim history | Label or derived |
| Cell (Wireless) | Call data records | Label column (`True/False`) |

The only hard requirement is a **CSV file with at least one numeric column**.

---

## 🧰 Tech Stack

### Core Language
- **Python 3.9+**

### Data Processing
| Library | Version | Purpose |
|---|---|---|
| `pandas` | ≥ 2.0.0 | DataFrame operations, ETL pipeline |
| `numpy` | ≥ 1.24.0 | Numerical computing, array operations |

### Machine Learning
| Library | Version | Purpose |
|---|---|---|
| `scikit-learn` | ≥ 1.3.0 | Random Forest, preprocessing, metrics, train/test split |
| `xgboost` | ≥ 2.0.0 *(optional)* | Gradient boosted trees — improves accuracy |

### Visualisation
| Library | Version | Purpose |
|---|---|---|
| `matplotlib` | ≥ 3.7.0 | Base chart rendering |
| `seaborn` | ≥ 0.12.0 | KDE plots, heatmaps, statistical charts |

### Optional Enhancements
| Library | Purpose |
|---|---|
| `shap` | SHAP feature-level explainability for individual predictions |

---

## ⚙️ Configuration

Edit `config.py` to tune model behaviour:

```python
# Model hyperparameters
RF_PARAMS = {
    "n_estimators":     300,      # More trees = more accurate but slower
    "class_weight":     "balanced",  # Handles imbalanced churn datasets
    "n_jobs":           1,        # Set to -1 to use all CPU cores (Linux/Windows)
}

XGB_PARAMS = {
    "n_estimators":     500,
    "learning_rate":    0.05,
    "max_depth":        6,
}

# Prediction threshold (default 50%)
CHURN_RISK_THRESHOLD = 0.5       # Lower = more sensitive, more churn flagged

# Split ratios
TEST_SIZE = 0.20                 # 20% held out for final evaluation
VAL_SIZE  = 0.10                 # 10% used for early stopping / tuning
```

---

## 📊 Output Files

After every run, the `outputs/` folder contains:

```
outputs/
├── churn_predictions_full.csv      # All customers: ID, probability, risk, advice
├── high_risk_customers.csv         # At-risk customers only, sorted by risk
├── 01_churn_overview.png
├── 02_risk_segments.png
├── 03_risk_distribution.png
├── 04_feature_importance.png
├── 05_model_comparison.png
├── 06_feature_distributions.png
├── 07_mean_comparison.png
├── 08_correlation_heatmap.png
└── ...  (additional charts based on your specific dataset)
```

**`churn_predictions_full.csv` columns:**
| Column | Description |
|---|---|
| `[id_column]` | Your original customer ID |
| `churn_probability` | Model confidence score (0.0–1.0) |
| `churn_prediction` | Binary prediction (1 = will churn) |
| `risk_level` | 🟢 Low / 🟡 Medium / 🟠 High / 🔴 Critical |
| `advice` | Human-readable recommended action |

---

## 🔧 Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'pandas'` | Run `source .venv/bin/activate` first |
| `FileNotFoundError: No such file or directory` | Place your CSV inside the `data/` subfolder |
| `ValueError: least populated class has only 1 member` | The churn derivation found very few churners — update `universal_preprocessor.py` (latest version fixes this) |
| `[mutex.cc] RAW: Lock blocking` warning on macOS | Harmless macOS threading warning — add `export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` before running |
| `XGBoost not installed` | Run `pip install xgboost` or use `--skip-xgb` flag |
| Churn rate shows 100% or 0% | The auto-derivation picked the wrong column — the system will auto-recover with a fallback strategy |

---

## 🗺️ Roadmap

- [ ] Streamlit web UI for drag-and-drop dataset upload
- [ ] SHAP waterfall plots for individual customer explanations
- [ ] LightGBM model integration
- [ ] Automated PDF report generation
- [ ] REST API endpoint for real-time scoring
- [ ] Support for Excel (`.xlsx`) and JSON input formats
- [ ] Multi-dataset comparison mode

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 👤 Author

**Alok**  
Built with Python, scikit-learn, and a lot of coffee ☕

---

*If this project helped you, please consider giving it a ⭐ on GitHub!*
