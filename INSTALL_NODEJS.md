🔧 **INSTALLING NODE.JS & RUNNING FRONTEND**

════════════════════════════════════════════════════════════════════════════

## ✅ Backend Status
Your backend IS running successfully on port 8001! 
✓ API available at http://localhost:8001
✓ API Docs at http://localhost:8001/docs

(The constant reloading was due to watching venv files - now fixed)

════════════════════════════════════════════════════════════════════════════

## 📦 Install Node.js

### Option 1: Using Homebrew (RECOMMENDED - Easiest)

```bash
# Install Homebrew if you don't have it:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Then install Node.js:
brew install node
```

### Option 2: Direct Download
Visit: https://nodejs.org/
- Download LTS version (20.x or 18.x)
- Run the installer
- Follow the prompts

### Verify Installation
```bash
node --version    # Should show v18.x or v20.x
npm --version     # Should show v9.x or v10.x
```

════════════════════════════════════════════════════════════════════════════

## 🚀 Start Frontend (After Node.js is installed)

Open a **NEW terminal** and run:

```bash
cd "/Users/alok/Desktop/Customer churn Predictor/frontend"
npm install
npm run dev
```

You should see:
```
▲ Next.js 14.0.0
- Local:        http://localhost:3000

✓ Ready in ...
```

════════════════════════════════════════════════════════════════════════════

## 🌐 Access the Application

Once both servers are running:

**Frontend:** http://localhost:3000
**Backend API:** http://localhost:8001
**Backend Docs:** http://localhost:8001/docs

════════════════════════════════════════════════════════════════════════════

## 💡 Quick Test

Without frontend, you can test backend API:

```bash
# Check if backend is working
curl http://localhost:8001/api/health

# Should return:
# {"status":"healthy","timestamp":"2026-05-08T...","xgboost_available":true}
```

════════════════════════════════════════════════════════════════════════════

**Next Step: Install Node.js, then run the frontend commands above**

