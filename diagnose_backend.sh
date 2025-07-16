#!/bin/bash

# MLBB Coach AI - Backend Diagnostic Script
# This script helps diagnose backend connection issues

echo "🔍 MLBB Coach AI Backend Diagnostics"
echo "====================================="
echo ""

# Check processes
echo "📊 1. Checking running processes..."
echo "Backend processes (uvicorn/FastAPI):"
ps aux | grep -E "(uvicorn|fastapi|app.py)" | grep -v grep || echo "   ❌ No backend processes found"
echo ""

# Check ports
echo "🌐 2. Checking port usage..."
echo "Port 8000 (Backend):"
lsof -i :8000 || echo "   ❌ Port 8000 not in use"
echo "Port 3000 (Frontend):"
lsof -i :3000 || echo "   ❌ Port 3000 not in use"
echo "Port 5173 (Vite Dev):"
lsof -i :5173 || echo "   ❌ Port 5173 not in use"
echo ""

# Test health endpoints
echo "🏥 3. Testing health endpoints..."
echo "Testing /api/health:"
if curl -s -f http://localhost:8000/api/health > /dev/null; then
    echo "   ✅ /api/health responding"
    curl -s http://localhost:8000/api/health | python3 -m json.tool 2>/dev/null || echo "   Response not JSON"
else
    echo "   ❌ /api/health not responding"
fi

echo "Testing /health:"
if curl -s -f http://localhost:8000/health > /dev/null; then
    echo "   ✅ /health responding"
    curl -s http://localhost:8000/health | python3 -m json.tool 2>/dev/null || echo "   Response not JSON"
else
    echo "   ❌ /health not responding"
fi

echo "Testing root endpoint:"
if curl -s -f http://localhost:8000/ > /dev/null; then
    echo "   ✅ Root endpoint responding"
else
    echo "   ❌ Root endpoint not responding"
fi
echo ""

# Check dependencies
echo "🔧 4. Checking Python dependencies..."
cd skillshift-ai 2>/dev/null || echo "❌ Cannot access skillshift-ai directory"

if [ -d "venv" ]; then
    echo "   ✅ Virtual environment exists"
    source venv/bin/activate 2>/dev/null || echo "   ❌ Cannot activate venv"
    
    echo "Checking key packages:"
    python3 -c "import fastapi; print('   ✅ FastAPI available')" 2>/dev/null || echo "   ❌ FastAPI not available"
    python3 -c "import uvicorn; print('   ✅ Uvicorn available')" 2>/dev/null || echo "   ❌ Uvicorn not available"
    python3 -c "from web.app import app; print('   ✅ App module imports successfully')" 2>/dev/null || echo "   ❌ App module import failed"
else
    echo "   ❌ Virtual environment not found"
fi
echo ""

# Network connectivity
echo "🌐 5. Network connectivity..."
echo "Localhost connectivity:"
if ping -c 1 localhost > /dev/null 2>&1; then
    echo "   ✅ Localhost reachable"
else
    echo "   ❌ Localhost unreachable"
fi
echo ""

# Recent logs (if available)
echo "📝 6. Recent activity..."
echo "Checking for recent uvicorn processes in history:"
history | grep uvicorn | tail -3 || echo "   No recent uvicorn commands in history"
echo ""

# Recommendations
echo "💡 7. Troubleshooting Recommendations:"
echo "=================================="

# Check if any process is running
if pgrep -f uvicorn > /dev/null; then
    echo "✅ Backend appears to be running"
    if ! curl -s -f http://localhost:8000/api/health > /dev/null; then
        echo "❌ But health endpoint not responding - possible startup issue"
        echo "💡 Try: Kill backend and restart with the startup script"
        echo "   killall -9 python3"
        echo "   ./start_services.sh"
    fi
else
    echo "❌ Backend not running"
    echo "💡 Start backend with: ./start_services.sh"
fi

echo ""
echo "🔧 Quick Fix Commands:"
echo "====================="
echo "# Kill all processes and restart:"
echo "killall -9 python3 node"
echo "cd skillshift-ai && ./start_services.sh"
echo ""
echo "# Manual backend start:"
echo "cd skillshift-ai"
echo "source venv/bin/activate"
echo "python3 -m uvicorn web.app:app --reload --port 8000"
echo ""
echo "# Test health manually:"
echo "curl http://localhost:8000/api/health"
echo "curl http://localhost:8000/health" 