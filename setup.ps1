# QUICK START SCRIPT FOR CHURN PREDICTION WEB APP (Windows)
# Run this in PowerShell with Administrator privileges

Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  🚀 Churn Prediction System - Quick Start Setup            ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Host "✓ Checking Python installation..." -ForegroundColor Green
python --version 2>$null || { Write-Host "❌ Python is required"; exit 1 }

# Check Node.js
Write-Host "✓ Checking Node.js installation..." -ForegroundColor Green
node --version 2>$null || { Write-Host "❌ Node.js is required"; exit 1 }

# Get project root
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommandPath
Write-Host "✓ Project root: $ProjectRoot" -ForegroundColor Green

# Setup Backend
Write-Host ""
Write-Host "📦 Setting up Backend..." -ForegroundColor Yellow
pip install -r "$ProjectRoot\backend\requirements.txt" -q
Write-Host "✓ Backend dependencies installed" -ForegroundColor Green

# Setup Frontend
Write-Host ""
Write-Host "📦 Setting up Frontend..." -ForegroundColor Yellow
Push-Location "$ProjectRoot\frontend"
npm install -q
Pop-Location
Write-Host "✓ Frontend dependencies installed" -ForegroundColor Green

# Success message
Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  ✅ Setup Complete! Ready to start servers                ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "📝 Next steps:" -ForegroundColor Cyan
Write-Host ""
Write-Host "   PowerShell 1 - Start Backend:" -ForegroundColor Yellow
Write-Host "   > cd '$ProjectRoot'" -ForegroundColor White
Write-Host "   > python backend/main.py" -ForegroundColor White
Write-Host ""
Write-Host "   PowerShell 2 - Start Frontend:" -ForegroundColor Yellow
Write-Host "   > cd '$ProjectRoot\frontend'" -ForegroundColor White
Write-Host "   > npm run dev" -ForegroundColor White
Write-Host ""
Write-Host "🌐 Once both are running, open:" -ForegroundColor Cyan
Write-Host "   http://localhost:3000" -ForegroundColor Green
Write-Host ""
Write-Host "📚 Full documentation: FULL_STACK_README.md" -ForegroundColor Cyan
Write-Host ""

