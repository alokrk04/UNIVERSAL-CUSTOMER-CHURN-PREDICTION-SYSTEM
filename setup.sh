#!/bin/bash

# QUICK START SCRIPT FOR CHURN PREDICTION WEB APP
# This script automates the setup process

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  🚀 Churn Prediction System - Quick Start Setup            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check Python
echo "✓ Checking Python installation..."
python3 --version || { echo "❌ Python 3 is required"; exit 1; }

# Check Node.js
echo "✓ Checking Node.js installation..."
node --version || { echo "❌ Node.js is required"; exit 1; }

# Get project root
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
echo "✓ Project root: $PROJECT_ROOT"

# Setup Backend
echo ""
echo "📦 Setting up Backend..."
pip install -r "$PROJECT_ROOT/backend/requirements.txt" --quiet
echo "✓ Backend dependencies installed"

# Setup Frontend
echo ""
echo "📦 Setting up Frontend..."
cd "$PROJECT_ROOT/frontend"
npm install --quiet
echo "✓ Frontend dependencies installed"

# Success message
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  ✅ Setup Complete! Ready to start servers                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📝 Next steps:"
echo ""
echo "   Terminal 1 - Start Backend:"
echo "   $ cd '$PROJECT_ROOT'"
echo "   $ python backend/main.py"
echo ""
echo "   Terminal 2 - Start Frontend:"
echo "   $ cd '$PROJECT_ROOT/frontend'"
echo "   $ npm run dev"
echo ""
echo "🌐 Once both are running, open:"
echo "   http://localhost:3000"
echo ""
echo "📚 Full documentation: FULL_STACK_README.md"
echo ""

