#!/usr/bin/env bash
# Start script for development (macOS/Linux)

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Churn Prediction System - Development Server Launcher    ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if tmux is installed for easier terminal management
if command -v tmux &> /dev/null; then
    echo "📌 Using tmux to manage terminals..."

    # Create new session
    tmux new-session -d -s churn

    # Backend window
    tmux new-window -t churn -n "backend"
    tmux send-keys -t churn:backend "cd '$PROJECT_ROOT' && python backend/main.py" C-m

    # Frontend window
    tmux new-window -t churn -n "frontend"
    tmux send-keys -t churn:frontend "cd '$PROJECT_ROOT/frontend' && npm run dev" C-m

    # Attach to session
    echo "✓ Servers started in tmux session 'churn'"
    echo "  - Attach: tmux attach-session -t churn"
    echo "  - Switch windows: Ctrl+B then L/R"
    echo ""
    tmux attach-session -t churn
else
    echo "⚠ tmux not found. Starting servers sequentially..."
    echo "You'll need to use separate terminal windows."
    echo ""
    echo "1. In Terminal 1:"
    echo "   cd '$PROJECT_ROOT'"
    echo "   python backend/main.py"
    echo ""
    echo "2. In Terminal 2:"
    echo "   cd '$PROJECT_ROOT/frontend'"
    echo "   npm run dev"
fi

