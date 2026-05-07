# 📐 ARCHITECTURE & DEVELOPMENT GUIDE

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER BROWSER                             │
│                    (http://localhost:3000)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │          REACT/NEXT.JS FRONTEND (Port 3000)              │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │  ┌─────────────┬──────────────┬──────────────┐           │  │
│  │  │   Page 1    │    Page 2    │    Page 3    │           │  │
│  │  │   Upload &  │              │              │           │  │
│  │  │   Overview  │ Visualizations│   Scoring    │           │  │
│  │  └─────────────┴──────────────┴──────────────┘           │  │
│  │                      ▲                                    │  │
│  │            (HTTP via StateContext)                        │  │
│  │                      │                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                         │                                       │
└─────────────────────────┼───────────────────────────────────────┘
                          │
                          │ HTTP/REST
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│          FASTAPI BACKEND (Port 8000)                            │
├─────────────────────────────────────────────────────────────────┤
│  /api/upload                                                    │
│  /api/run-pipeline ──────┐                                      │
│  /api/status             │                                      │
│  /api/visualizations     │ ●● Background Task Processing        │
│  /api/predictions/*      │ (Threading)                          │
│  /api/image/*            │                                      │
│                          └──────────┐                           │
│                                      ▼                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │      CHURN PREDICTION PIPELINE (Background)              │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │  1. Auto-Detect  ──▶  2. ETL  ──▶  3. Features           │  │
│  │                                                          │  │
│  │  4. Train Models ──▶  5. Generate Viz. ──▶ 6. Export    │  │
│  │      (RF/XGB)           (PNG)              (CSV)         │  │
│  │                                                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ▲                                      │
│        (Imports from existing Python modules)                   │
│                          │                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │      EXISTING CHURN SYSTEM MODULES                       │  │
│  │  ├─ auto_detector.py                                    │  │
│  │  ├─ universal_preprocessor.py                           │  │
│  │  ├─ universal_features.py                               │  │
│  │  ├─ churn_models.py                                     │  │
│  │  ├─ predictor.py                                        │  │
│  │  ├─ universal_visualizer.py                             │  │
│  │  └─ config.py                                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                          ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │            DATA & MODELS STORAGE                         │  │
│  │  ├─ /data/         (Uploaded CSV files)                 │  │
│  │  ├─ /outputs/      (Predictions & visualizations)       │  │
│  │  └─ /models/       (Trained ML models)                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Frontend Architecture

### Component Hierarchy

```
app/
├── layout.jsx (Root Layout)
│   └── StateProvider (Context wrapper)
│       ├── Sidebar (Navigation)
│       └── {children}
│           ├── page.jsx (Upload & Overview)
│           ├── visualizations/page.jsx (Charts)
│           └── scoring/page.jsx (Predictions)
│
context/
└── StateContext.jsx
    ├── state (Pipeline status & results)
    ├── predictions (All customer predictions)
    ├── highRiskPredictions (Flagged customers)
    ├── visualizations (Image metadata)
    └── Helper functions
        ├── uploadFile()
        ├── runPipeline()
        ├── fetchPredictions()
        ├── fetchVisualizations()
        └── resetPipeline()
```

### State Flow Diagram

```
Initial State
    │
    ├─ User clicks upload
    │  └─ uploadFile() 
    │     └─ POST /api/upload
    │        └─ state.status = 'uploaded'
    │
    ├─ User clicks "Run Pipeline"
    │  └─ runPipeline()
    │     └─ POST /api/run-pipeline
    │        └─ state.status = 'running'
    │
    ├─ Frontend polls /api/status every 1s
    │  └─ Updates state.progress, state.message
    │
    ├─ Pipeline completes
    │  └─ state.status = 'complete'
    │     └─ state.results populated
    │
    ├─ User navigates to pages
    │  ├─ Visualizations page
    │  │  └─ fetchVisualizations()
    │  │     └─ GET /api/visualizations
    │  │
    │  └─ Scoring page
    │     └─ fetchPredictions()
    │        └─ GET /api/predictions/all
    │
    └─ User clicks reset
       └─ resetPipeline()
          └─ POST /api/reset
             └─ state = initial
```

### Styling Strategy

- **Tailwind CSS**: Utility-first CSS framework
- **Component Classes**: 
  - `.btn-primary`, `.btn-secondary` for buttons
  - `.card` for container styling
  - `.badge-*` for risk level badges
- **Responsive Design**:
  - Mobile-first approach
  - `lg:` breakpoint for desktop
  - Sidebar collapses on mobile

---

## Backend Architecture

### Request/Response Flow

```
Frontend HTTP Request
    │
    ▼
FastAPI Route Handler
    ├─ Validation (file type, size, etc.)
    ├─ Error handling
    │
    ├─ If POST /api/upload:
    │  ├─ Save file to disk
    │  └─ Return file metadata
    │
    ├─ If POST /api/run-pipeline:
    │  ├─ Check if file uploaded
    │  ├─ Launch background thread
    │  │  └─ run_pipeline_background()
    │  └─ Return "pipeline started" immediately
    │
    ├─ If GET /api/status:
    │  ├─ Return current pipeline state
    │  └─ (Frontend polls this every 1s)
    │
    └─ If GET /api/predictions/*:
       ├─ Read CSV from outputs/
       ├─ Convert to JSON
       └─ Return to frontend
    
    ▼
Response JSON
    │
    ▼
Frontend State Updated
```

### Pipeline Execution (Background Thread)

```python
def run_pipeline_background(csv_path, dataset_name, skip_xgb):
    """
    Executes in separate thread - doesn't block API
    """
    try:
        # Phase 1: Load & Detect
        update_state(progress=15)
        df_clean, y, df_raw, schema = run_universal_etl(csv_path)
        
        # Phase 2: Transform
        update_state(progress=30)
        df_feat = engineer_features(df_clean, schema, df_raw)
        df_feat = select_features(df_feat, y, max_features=150)
        
        # Phase 3: Split
        update_state(progress=45)
        X_train, X_val, X_test, ... = split_dataset(df_feat, y)
        
        # Phase 4: Train Models (50% -> 75%)
        update_state(progress=50)
        rf = RandomForestChurnModel().fit(...).evaluate(...)
        
        if has_xgb and not skip_xgb:
            update_state(progress=65)
            xgb = XGBoostChurnModel().fit(...).evaluate(...)
        
        # Phase 5: Ensemble (75%)
        update_state(progress=75)
        ensemble = EnsembleChurnModel(rf=rf, xgb=xgb, ...)
        
        # Phase 6: Visualize (80%)
        update_state(progress=80)
        run_all_visualizations(...)
        
        # Phase 7: Export (85% -> 100%)
        update_state(progress=85)
        predictions_df.to_csv(...)
        high_risk_df.to_csv(...)
        
        # Phase 8: Finalize
        update_state(status='complete', progress=100)
        
    except Exception as e:
        update_state(status='error', error=str(e))
```

### State Management (Thread-Safe)

```python
PIPELINE_STATE = {
    "status": "idle",
    "progress": 0,
    "message": "",
    "dataset_name": "",
    "csv_path": "",
    "results": None,
    "schema": None,
    "error": None,
    "duration": 0,
}

STATE_LOCK = threading.Lock()  # Prevents race conditions

def update_state(status=None, progress=None, **kwargs):
    with STATE_LOCK:
        # Thread-safe updates
        PIPELINE_STATE.update({...})
```

---

## Data Flow Examples

### 1. File Upload Flow

```
User selects CSV
    │
    ▼
Frontend: onDrop() → uploadFile()
    │
    ├─ Create FormData with file
    └─ POST to /api/upload
        │
        ▼
Backend: @app.post("/api/upload")
    │
    ├─ Validate file is CSV
    ├─ Save to /backend/uploads/
    ├─ Read with pd.read_csv()
    ├─ Validate not empty
    └─ Return {filename, rows, columns}
        │
        ▼
Frontend: setState({status: 'uploaded'})
    │
    └─ Display "File uploaded successfully"
```

### 2. Pipeline Execution Flow

```
User clicks "Run Pipeline"
    │
    ▼
Frontend: runPipeline()
    │
    ├─ POST to /api/run-pipeline
    └─ setState({status: 'running'})
        │
        ▼
Backend: @app.post("/api/run-pipeline")
    │
    ├─ Validate CSV uploaded
    ├─ setState({status: 'running'})
    ├─ Start background thread
    │  └─ run_pipeline_background()
    └─ Return immediately
        │
        ▼
Frontend: Start polling /api/status
    │
    ├─ Every 1 second:
    │  └─ GET /api/status
    │     └─ setState(data)
    │        └─ Update progress bar
    │
    └─ When status='complete':
       ├─ Stop polling
       └─ Display results summary
```

### 3. Predictions Display Flow

```
User navigates to Scoring page
    │
    ▼
Frontend: useEffect(() => fetchPredictions())
    │
    ├─ GET /api/predictions/all
    └─ GET /api/predictions/high-risk
        │
        ▼
Backend: Read from outputs/
    │
    ├─ churn_predictions_full.csv
    ├─ high_risk_customers.csv
    └─ Return as JSON
        │
        ▼
Frontend: 
    │
    ├─ Filter by search query
    ├─ Filter by risk level
    └─ Render data table
        │
        └─ User can:
           ├─ Search by ID
           ├─ Filter by risk
           ├─ View full advice
           └─ Download CSV
```

---

## Database/Storage Structure

```
Project Root/
├── backend/
│   └── uploads/                    (Temporary uploaded files)
│       └── user_dataset_123.csv
│
├── data/                           (Input data directory)
│   └── [auto-populated from uploads]
│
├── outputs/                        (Generated results)
│   ├── 01_churn_overview.png
│   ├── 02_risk_segments.png
│   ├── ... (10 PNG files)
│   ├── churn_predictions_full.csv  (All customers)
│   └── high_risk_customers.csv     (Flagged customers)
│
└── models/                         (Trained ML models)
    ├── rf_model.pkl                (Random Forest)
    └── xgb_model.pkl               (XGBoost)
```

---

## Key Technologies & Why

| Layer | Technology | Why |
|-------|-----------|-----|
| **Frontend** | Next.js 14 | Server components, SSR, built-in optimization |
| | React 18 | Component-based UI, hooks, context API |
| | Tailwind CSS | Rapid UI development, responsive design |
| | Lucide React | Consistent, lightweight icons |
| **Backend** | FastAPI | High performance, automatic API docs |
| | Python 3.11 | Existing system uses Python |
| | Uvicorn | ASGI server, concurrent requests |
| **Communication** | HTTP/REST | Simple, stateless, widely supported |
| **State** | React Context | No external dependencies, simple state |
| | Threading (Python) | Non-blocking background tasks |
| **Styling** | Tailwind CSS | Utility-first, responsive, fast |

---

## Development Workflow

### Local Development

```bash
# 1. Make code changes
# 2. Frontend auto-reloads (npm run dev)
# 3. Backend auto-reloads (uvicorn --reload)
# 4. Test in browser

# Typical workflow:
# Terminal 1: python backend/main.py  (auto-reload)
# Terminal 2: cd frontend && npm run dev
# Terminal 3: Make code changes in editor
```

### Testing the API

```bash
# Using curl
curl http://localhost:8000/api/health

# Using VS Code REST Client
# Create file: requests.http
GET http://localhost:8000/api/health

# Using Swagger UI
open http://localhost:8000/docs
```

### Debugging

**Frontend (DevTools)**
```javascript
// In browser console
// Access state from window
console.log(window.__NEXT_DATA__)

// Network tab shows all API calls
// Application tab shows localStorage/cookies
```

**Backend (Python)**
```python
# Add print statements or use debugger
import pdb; pdb.set_trace()

# Or use pytest
pytest backend/test_api.py -v
```

---

## Performance Optimization

### Frontend
- ✅ Code splitting (Next.js automatic)
- ✅ Image optimization
- ✅ CSS minification (Tailwind)
- ✅ React.memo for heavy components
- ✅ Debounced search

### Backend
- ✅ Background task processing
- ✅ Response caching
- ✅ Database query optimization
- ✅ Connection pooling
- ✅ Gzip compression

---

## Security Measures

```
✅ Frontend:
  - No sensitive data in localStorage
  - CSRF token handling (if needed)
  - XSS protection via React escaping

✅ Backend:
  - Input validation (file type, size)
  - SQL injection prevention (N/A, no DB)
  - CORS configured for localhost:3000
  - HTTPS in production (via proxy)
  - Rate limiting (can be added)

✅ Communication:
  - HTTPS in production
  - No credentials in URLs
  - Auth headers (if needed)
```

---

## Scaling Considerations

```
Current Architecture → Production Scaling

Frontend:
  Next.js Dev Server  →  Vercel / Netlify / CDN
  Tailwind CSS        →  Remains same
  Context API         →  Redux / Zustand (if needed)

Backend:
  Single Unicorn      →  Multiple workers + nginx
  Thread-based        →  Celery + Redis for queues
  Local files         →  S3 / Cloud storage
  Local DB            →  PostgreSQL / MongoDB
  localhost:8000      →  Health checks + load balancer
```

---

## Troubleshooting Guide

| Problem | Diagnosis | Solution |
|---------|-----------|----------|
| Port already in use | `lsof -i :8000` | Kill process or use different port |
| CORS error | Browser console | Check backend CORS config |
| Slow pipeline | Monitor backend | Increase workers, reduce dataset size |
| Memory error | System monitor | Reduce max_features, sample data |
| Stale data | Check API response | Clear browser cache + F5 |

---

**For detailed development instructions, see: FULL_STACK_README.md**

**For quick start, see: QUICK_START.md**

