#!/bin/bash

# SkillShift AI - Complete Service Startup Script with Redis Queue Worker
# This script starts the frontend, backend, and Redis worker services

echo "🚀 Starting SkillShift AI Services with Async Processing..."

# Check if we're in the right directory
if [ ! -f "web/app.py" ]; then
    echo "❌ Error: Please run this script from the skillshift-ai directory"
    exit 1
fi

# Function to check if a port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        echo "⚠️  Port $1 is already in use"
        return 0
    else
        return 1
    fi
}

# Function to check if Redis is running
check_redis() {
    # Check if using Redis Cloud (REDIS_URL set) or local Redis
    if [ -n "$REDIS_URL" ]; then
        echo "🌐 Using Redis Cloud configuration"
        echo "   Redis URL: ${REDIS_URL%@*}@***" # Hide password in logs
        return 0
    else
        echo "🔧 Using local Redis configuration"
        if redis-cli ping >/dev/null 2>&1; then
            echo "✅ Local Redis is running"
            return 0
        else
            echo "❌ Local Redis is not running"
            return 1
        fi
    fi
}

# Check and start Redis if needed
echo "🔧 Checking Redis configuration..."
if ! check_redis; then
    # Only try to start local Redis if REDIS_URL is not set
    if [ -z "$REDIS_URL" ]; then
        echo "🚀 Starting local Redis service..."
        if command -v brew >/dev/null 2>&1; then
            brew services start redis
            sleep 2
            if ! check_redis; then
                echo "❌ Failed to start Redis via Homebrew"
                echo "Please start Redis manually: brew services start redis"
                echo "Or set REDIS_URL environment variable for Redis Cloud"
                exit 1
            fi
        else
            echo "❌ Homebrew not found. Please either:"
            echo "   1. Start Redis manually (macOS: brew services start redis)"
            echo "   2. Set REDIS_URL environment variable for Redis Cloud"
            exit 1
        fi
    else
        echo "✅ Redis Cloud URL configured, skipping local Redis check"
    fi
fi

# Kill any existing processes on the ports
echo "🧹 Cleaning up existing processes..."
if check_port 8000; then
    echo "Stopping existing backend on port 8000..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
fi

if check_port 3000; then
    echo "Stopping existing frontend on port 3000..."
    lsof -ti:3000 | xargs kill -9 2>/dev/null || true
fi

# Stop any existing workers
echo "🧹 Stopping existing Redis workers..."
pkill -f "python worker.py" 2>/dev/null || true
pkill -f "rq worker" 2>/dev/null || true

# Wait a moment for ports to be released
sleep 2

# Start Redis worker in background
echo "⚙️  Starting Redis analysis worker..."
cd "$(dirname "$0")"
python worker.py &
WORKER_PID=$!

# Wait for worker to initialize
echo "⏳ Waiting for worker to initialize..."
sleep 3

# Start backend in background
echo "🔧 Starting FastAPI backend on port 8000..."
python -m uvicorn web.app:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Wait for backend to start
echo "⏳ Waiting for backend to initialize..."
sleep 5

# Check if backend is healthy
if curl -s http://localhost:8000/health >/dev/null; then
    echo "✅ Backend is healthy"
else
    echo "❌ Backend failed to start"
    kill $BACKEND_PID $WORKER_PID 2>/dev/null
    exit 1
fi

# Start frontend in background
echo "🎨 Starting Next.js frontend on port 3000..."
cd dashboard-ui
npm run dev &
FRONTEND_PID=$!

# Wait for frontend to start
echo "⏳ Waiting for frontend to initialize..."
sleep 8

# Check if frontend is healthy
if curl -s http://localhost:3000 >/dev/null; then
    echo "✅ Frontend is healthy"
else
    echo "❌ Frontend failed to start"
    kill $BACKEND_PID $FRONTEND_PID $WORKER_PID 2>/dev/null
    exit 1
fi

echo ""
echo "🎉 All services started successfully!"
echo ""
echo "📊 Access points:"
echo "   Frontend Dashboard: http://localhost:3000"
echo "   Backend API:        http://localhost:8000"
echo "   Health Check:       http://localhost:3000/api/health"
echo ""
echo "🔄 Async Processing:"
if [ -n "$REDIS_URL" ]; then
    echo "   Redis Cloud:        Connected (${REDIS_URL%@*}@***)"
else
    echo "   Local Redis:        Running on localhost:6379"
fi
echo "   Worker:             Active (PID: $WORKER_PID)"
echo "   Job Queue:          'analysis' queue"
echo "   Async Endpoint:     /api/analyze-async"
echo ""
echo "Press Ctrl+C to stop all services"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping all services..."
    kill $BACKEND_PID $FRONTEND_PID $WORKER_PID 2>/dev/null
    echo "✅ All services stopped"
    exit 0
}

# Set trap to cleanup on exit
trap cleanup SIGINT SIGTERM

# Wait for user to stop services
wait 