🚀 **QUICK START GUIDE**

# 30-Second Setup (macOS/Linux)

```bash
# 1. Install dependencies
bash setup.sh

# 2. Terminal 1 - Start Backend
python backend/main.py

# 3. Terminal 2 - Start Frontend
cd frontend && npm run dev

# 4. Open http://localhost:3000
```

---

# 30-Second Setup (Windows)

```powershell
# 1. Install dependencies (Run as Administrator)
.\setup.ps1

# 2. PowerShell 1 - Start Backend
python backend/main.py

# 3. PowerShell 2 - Start Frontend
cd frontend
npm run dev

# 4. Open http://localhost:3000
```

---

# Docker Setup (All Platforms)

```bash
# Build and start all services
docker-compose up --build

# The app will be available at:
# Frontend: http://localhost:3000
# Backend: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

---

# Manual Setup

## Backend Setup

```bash
# Navigate to project
cd /Users/alok/Desktop/"Customer churn Predictor"

# Create virtual environment (optional)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Start server
python backend/main.py

# Expected output:
# ╔══════════════════════════════════════════════════════════════════╗
# ║        CHURN PREDICTION SYSTEM - FastAPI Backend                 ║
# ║        Starting server on http://localhost:8000                   ║
# ║        API docs available at http://localhost:8000/docs          ║
# ╚══════════════════════════════════════════════════════════════════╝
```

## Frontend Setup

```bash
# In a new terminal
cd /Users/alok/Desktop/"Customer churn Predictor"/frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Expected output:
# ▲ Next.js 14.0.0
#   - Local:        http://localhost:3000
#   - Environments: .env.local
# 
# ✓ Ready in 2.5s
```

---

# 🎯 First Run Walkthrough

1. **Open http://localhost:3000**
   - You'll see the Upload & Overview page

2. **Upload a CSV**
   - Click the upload zone or drag-and-drop
   - Try one of the example files:
     - `/data/telecom_churn_data.csv`
     - `CSV's/BankChurners.csv`
     - `CSV's/patient_churn_dataset.csv`

3. **Click "Run Churn Pipeline"**
   - Watch the progress bar fill up
   - Takes 30-120 seconds depending on dataset size

4. **View Results**
   - Check the Summary panel on the right
   - Navigate to Visualizations page to see charts
   - Go to Scoring Center to view predictions

---

# 📊 Example Prediction CSV

Once you run the pipeline, you'll see predictions like:

| Customer ID | Churn Probability | Risk Level | Advice |
|-------------|-------------------|-----------|---------|
| CUST001 | 82.5% | 🔴 Critical | Imminent churn — immediate intervention required |
| CUST002 | 45.3% | 🟡 Medium | Some risk signals — consider a retention offer |
| CUST003 | 12.1% | 🟢 Low | Customer is active and unlikely to churn |

---

# 🔧 Common Commands

### Backend
```bash
# Start with auto-reload (development)
cd backend
uvicorn main:app --reload

# View API documentation
http://localhost:8000/docs

# Check backend health
curl http://localhost:8000/api/health
```

### Frontend
```bash
# Development with hot reload
cd frontend && npm run dev

# Build for production
cd frontend && npm run build

# Start production build
cd frontend && npm start

# Linting
cd frontend && npm run lint
```

### Full App Reset
```bash
# Reset pipeline state and clear uploads
curl -X POST http://localhost:8000/api/reset
```

---

# ⚡ Performance Tips

1. **Faster Pipeline**
   - Use smaller datasets (< 100K rows)
   - Skip XGBoost: Add `?skip_xgb=true` to pipeline URL
   - Use telecom_churn_data.csv for testing (included)

2. **Faster Frontend**
   - Use Chrome/Safari for best performance
   - Frontend caches visualizations automatically
   - Clear browser cache: Ctrl+Shift+Delete

3. **Faster Backend**
   - Increase workers: `uvicorn main:app --workers 4`
   - Use production mode: `DEBUG=False`
   - Enable caching for predictions

---

# 🐛 Troubleshooting

**Backend won't start?**
```bash
# Check if port 8000 is in use
lsof -i :8000

# Kill existing process
kill -9 <PID>
```

**Frontend won't load?**
```bash
# Check if port 3000 is in use
lsof -i :3000

# If using different port, update .env.local:
# NEXT_PUBLIC_API_URL=http://localhost:8000
```

**CORS errors?**
- Backend CORS is already configured for localhost:3000
- For other origins, edit `backend/main.py` line with `CORSMiddleware`

**Out of memory?**
- Use a smaller dataset
- Reduce max_features in config.py
- Close other applications

---

# 📚 Full Documentation

For complete instructions, troubleshooting, API reference, and advanced configuration:

👉 **See: FULL_STACK_README.md**

---

**You're all set! Happy predicting! 🎯**

