# 🎯 Universal Customer Churn Prediction System - Full-Stack Web Application

A modern, reactive web application for customer churn prediction built with **FastAPI** (backend) and **Next.js + React** (frontend).

## 🚀 Features

### Backend (FastAPI)
- ✅ RESTful API for file upload and pipeline execution
- ✅ Background task processing for long-running ML pipeline
- ✅ Real-time status updates and progress tracking
- ✅ Image serving for generated visualizations
- ✅ CSV data export and streaming
- ✅ Advanced search and filtering capabilities
- ✅ CORS enabled for frontend integration
- ✅ Comprehensive error handling

### Frontend (Next.js + React)
- ✅ **Page 1 - Data Upload & Overview**: Drag-and-drop CSV upload with auto-schema detection summary
- ✅ **Page 2 - Visualizations**: Interactive gallery of generated charts and heatmaps
- ✅ **Page 3 - Customer Scoring**: Data table with search, filtering, and risk-level badges
- ✅ Modern UI with Tailwind CSS and Lucide React icons
- ✅ Responsive design (mobile, tablet, desktop)
- ✅ Real-time progress tracking with visual feedback
- ✅ State management with React Context
- ✅ Automatic polling for pipeline status

## 📋 System Requirements

- Python 3.8+
- Node.js 18+ (LTS)
- npm or yarn
- 4GB RAM minimum
- Access to localhost (ports 3000 and 8000)

## 📁 Project Structure

```
Customer churn Predictor/
├── backend/
│   ├── main.py                 # FastAPI server
│   ├── requirements.txt        # Python dependencies
│   ├── .env                    # Environment configuration
│   └── uploads/                # Uploaded CSV files (auto-created)
│
├── frontend/
│   ├── app/
│   │   ├── layout.jsx         # Root layout with Context provider
│   │   ├── page.jsx           # Home page (upload & overview)
│   │   ├── visualizations/
│   │   │   └── page.jsx       # Visualizations page
│   │   └── scoring/
│   │       └── page.jsx       # Scoring center page
│   │
│   ├── components/
│   │   └── Sidebar.jsx        # Navigation sidebar
│   │
│   ├── context/
│   │   └── StateContext.jsx   # Global state management
│   │
│   ├── styles/
│   │   └── globals.css        # Global Tailwind styles
│   │
│   ├── package.json
│   ├── next.config.js
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── .env.local             # Environment configuration
│   └── .gitignore
│
├── (Original Python modules)
├── config.py
├── auto_detector.py
├── universal_preprocessor.py
├── universal_features.py
├── churn_models.py
├── predictor.py
├── universal_visualizer.py
└── main.py                    # Original CLI entry point
```

## 🔧 Installation

### Step 1: Clone/Navigate to Project

```bash
cd /Users/alok/Desktop/"Customer churn Predictor"
```

### Step 2: Setup Backend

#### 2.1 Create Virtual Environment (Optional but Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 2.2 Install Backend Dependencies

```bash
pip install -r backend/requirements.txt
```

#### 2.3 Verify Installation

```bash
python -c "import fastapi, pandas, sklearn; print('✅ All dependencies installed!')"
```

### Step 3: Setup Frontend

#### 3.1 Navigate to Frontend Directory

```bash
cd frontend
```

#### 3.2 Install Node Dependencies

```bash
npm install
# or
yarn install
```

#### 3.3 Verify Installation

```bash
npm run build
```

## 🏃 Running the Application

### Development Mode (Recommended)

#### Terminal 1: Start FastAPI Backend

```bash
# From project root directory
python backend/main.py

# Or using uvicorn directly:
cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Expected output:
```
╔══════════════════════════════════════════════════════════════════╗
║        CHURN PREDICTION SYSTEM - FastAPI Backend                 ║
║        Starting server on http://localhost:8000                   ║
║        API docs available at http://localhost:8000/docs          ║
╚══════════════════════════════════════════════════════════════════╝
```

#### Terminal 2: Start Next.js Frontend

```bash
# From frontend directory
cd frontend
npm run dev

# Or if you're in the project root:
cd frontend && npm run dev
```

Expected output:
```
  ▲ Next.js 14.0.0
  - Local:        http://localhost:3000
  - Environments: .env.local

✓ Ready in 2.5s
```

#### Access the Application

Open your browser and navigate to:
- **Frontend**: http://localhost:3000
- **Backend API Docs**: http://localhost:8000/docs
- **Backend API**: http://localhost:8000

### Production Mode

#### Build Frontend

```bash
cd frontend
npm run build
npm start
```

#### Start Backend

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 📊 Application Walkthrough

### Page 1: Data Upload & Overview

1. **Upload CSV**: Drag-and-drop or click to select your dataset
2. **Auto-Schema Detection**: System automatically detects columns
3. **Run Pipeline**: Click "Run Churn Pipeline" to start analysis
4. **View Summary**: See results including:
   - Total customers and churn rate
   - Best model AUC score
   - Models trained (Random Forest, XGBoost, Ensemble)
   - Detected schema details

### Page 2: Visualizations & Insights

1. **View Charts**: Browse auto-generated visualizations in a responsive grid
2. **Click to Expand**: Click any chart for full-screen view
3. **Included Charts**:
   - Churn overview and distribution
   - Feature importance ranking
   - Risk segments and distribution
   - Correlation heatmaps
   - Usage and revenue trends
   - Model comparison metrics

### Page 3: Customer Scoring & Action Center

1. **Search**: Find customers by ID or any field
2. **Filter**: Filter by risk level (Low, Medium, High, Critical)
3. **Toggle View**: Switch between all customers and high-risk only
4. **View Advice**: Click "View Advice" for retention recommendations
5. **Download**: Export predictions as CSV

## 🔌 API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/upload` | Upload CSV file |
| POST | `/api/run-pipeline` | Start churn prediction pipeline |
| GET | `/api/status` | Get current pipeline status |
| GET | `/api/results` | Get detailed results |
| POST | `/api/reset` | Reset pipeline state |

### Visualization Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/visualizations` | List all generated images |
| GET | `/api/image/{filename}` | Serve image file |

### Prediction Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/predictions/all` | Get all predictions as JSON |
| GET | `/api/predictions/high-risk` | Get high-risk customers |
| GET | `/api/predictions/search?query=...` | Search predictions |
| GET | `/api/predictions/download?file_type=...` | Download predictions as CSV |

## 🧠 How It Works

### Pipeline Execution Flow

```
1. User uploads CSV
   ↓
2. Backend stores file → Submits to upload API
   ↓
3. Frontend displays upload confirmation
   ↓
4. User clicks "Run Pipeline"
   ↓
5. Backend launches background task:
   - Auto-detects schema
   - Runs ETL & feature engineering
   - Trains Random Forest & XGBoost
   - Creates Ensemble model
   - Generates visualizations
   - Exports predictions
   ↓
6. Frontend polls status endpoint every 1 second
   ↓
7. Progress bar updates in real-time
   ↓
8. When complete (status='complete'):
   - Summary displays on home page
   - Visualizations available on page 2
   - Customer predictions on page 3
```

### State Management

The app uses React Context API for global state management:

```javascript
{
  status: 'idle' | 'uploading' | 'running' | 'complete' | 'error',
  progress: 0-100,
  message: string,
  dataset_name: string,
  results: { ... },
  schema: { ... },
  error: string | null,
  duration: number,
}
```

## ⚙️ Configuration

### Backend Configuration (`backend/.env`)

```env
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False
```

Also edit `config.py` in the project root for ML model parameters:

```python
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 5,
    ...
}

XGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    ...
}
```

### Frontend Configuration (`frontend/.env.local`)

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 🐛 Troubleshooting

### Issue: "Connection refused" on localhost:8000

**Solution**: Make sure backend is running:
```bash
python backend/main.py
```

### Issue: Frontend won't load on localhost:3000

**Solution**: Make sure frontend dev server is running:
```bash
cd frontend && npm run dev
```

### Issue: CORS errors in browser console

**Solution**: Backend is configured to accept requests from localhost:3000. If running on a different URL, update `app.add_middleware(CORSMiddleware, ...)` in `backend/main.py`:

```python
allow_origins=["http://your-domain:port", "*"],
```

### Issue: Pipeline takes too long

**Solution**: 
- Increase test dataset size (use top N rows for testing)
- Skip XGBoost: pass `?skip_xgb=true` when running pipeline
- Use GPU if available (XGBoost supports CUDA)

### Issue: Out of memory errors

**Solution**:
- Reduce dataset size
- Lower `max_features` in `universal_features.py` (default: 150)
- Use sampling for large datasets

## 📝 Example CSV Formats Supported

The system auto-detects and handles:

### Telecom Dataset
```csv
Customer ID, Age, Tenure, Monthly Charges, Total Charges, Churn
```

### Bank Dataset
```csv
RowNumber, CustomerId, Surname, CreditScore, Geography, Attrition
```

### E-commerce Dataset
```csv
CustomerID, SignupDate, LastPurchaseDate, TotalSpent, OrderCount, Status
```

## 🚢 Deployment

### Docker (Recommended)

Create `Dockerfile` in project root:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY backend/requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t churn-predictor .
docker run -p 8000:8000 churn-predictor
```

### Heroku Deployment

1. Push to Git
2. Create `Procfile`:
```
web: python -m uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```
3. Deploy: `git push heroku main`

## 📞 Support & Debugging

### Check API Health

```bash
curl http://localhost:8000/api/health
```

### View API Documentation

Visit: http://localhost:8000/docs (Swagger UI)
or: http://localhost:8000/redoc (ReDoc)

### Check Logs

Backend logs appear in terminal where you ran `python backend/main.py`

Frontend logs appear in browser console (F12)

## 🔐 Security Considerations

- ✅ CORS is restricted to localhost:3000 in development
- ✅ File uploads are validated (CSV only)
- ✅ No sensitive data stored in frontend state
- ✅ API runs on localhost by default
- ⚠️ For production: Set proper CORS origins, use HTTPS, add authentication

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Next.js Documentation](https://nextjs.org/docs)
- [Tailwind CSS](https://tailwindcss.com/)
- [React Context API](https://react.dev/reference/react/useContext)

## 📄 License

This project maintains the same license as the original Universal Customer Churn Prediction System.

---

**Happy Predicting! 🚀**

For issues or questions, check the troubleshooting section above or review the API docs at http://localhost:8000/docs.

