#!/bin/bash

# MLBB Coach AI - Service Startup Script
# This script starts both backend and frontend services with proper health checking

echo "🚀 Starting MLBB Coach AI Services..."
echo "=================================="

# Kill existing processes on ports
echo "🧹 Cleaning up existing processes..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true
lsof -ti:5173 | xargs kill -9 2>/dev/null || true

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/update requirements
echo "📋 Installing Python dependencies..."
pip install -r requirements.txt --quiet

# Start backend
echo "🖥️  Starting FastAPI backend..."
python3 -m uvicorn web.app:app --reload --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
echo "⏳ Waiting for backend to initialize..."
sleep 5

# Test backend health
echo "🔍 Testing backend health..."
for i in {1..10}; do
    if curl -s http://localhost:8000/api/health > /dev/null; then
        echo "✅ Backend is healthy! Health endpoint responding."
        break
    elif curl -s http://localhost:8000/health > /dev/null; then
        echo "✅ Backend is healthy! Fallback health endpoint responding."
        break
    else
        echo "⏳ Backend not ready yet... ($i/10)"
        sleep 2
    fi
    
    if [ $i -eq 10 ]; then
        echo "❌ Backend health check failed after 20 seconds"
        echo "🔍 Checking backend logs..."
        ps aux | grep uvicorn
        exit 1
    fi
done

# Start frontend
echo "🌐 Starting Next.js frontend..."
cd dashboard-ui
npm install --silent
npm run dev &
FRONTEND_PID=$!

echo ""
echo "🎉 Services Started Successfully!"
echo "=================================="
echo "🖥️  Backend:  http://localhost:8000"
echo "🌐 Frontend: http://localhost:3000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "💡 Health Endpoints:"
echo "   - http://localhost:8000/api/health"
echo "   - http://localhost:8000/health"
echo ""
echo "🛑 To stop services:"
echo "   kill $BACKEND_PID $FRONTEND_PID"
echo ""
echo "📊 Service Status:"
echo "   Backend PID: $BACKEND_PID"
echo "   Frontend PID: $FRONTEND_PID"
echo ""

# Monitor services
echo "🔍 Monitoring services (Ctrl+C to stop)..."
wait 