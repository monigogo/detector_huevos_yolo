#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if backend is running
is_backend_running() {
    pgrep -f "uvicorn src.demo.backend.main:app" > /dev/null
}

# Check if frontend is running
is_frontend_running() {
    pgrep -f "streamlit run src/demo/frontend/app.py" > /dev/null
}

# Start backend
start_backend() {
    if is_backend_running; then
        echo -e "${YELLOW}Backend is already running.${NC}"
    else
        echo -e "${GREEN}Starting FastAPI backend...${NC}"
        uvicorn src.demo.backend.main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
        sleep 2
        if is_backend_running; then
            echo -e "${GREEN}FastAPI backend started successfully.${NC}"
            echo -e "${GREEN}Backend API available at: http://localhost:8000${NC}"
        else
            echo -e "${RED}Failed to start FastAPI backend.${NC}"
        fi
    fi
}

# Start frontend
start_frontend() {
    if is_frontend_running; then
        echo -e "${YELLOW}Frontend is already running.${NC}"
    else
        echo -e "${GREEN}Starting frontend...${NC}"
        streamlit run src/demo/frontend/app.py > frontend.log 2>&1 &
        sleep 2
        if is_frontend_running; then
            echo -e "${GREEN}Frontend started successfully.${NC}"
            echo -e "${GREEN}You can now access the app at: http://localhost:8501${NC}"
        else
            echo -e "${RED}Failed to start frontend.${NC}"
        fi
    fi
}

# Stop backend
stop_backend() {
    if is_backend_running; then
        echo -e "${GREEN}Stopping FastAPI backend...${NC}"
        pkill -f "uvicorn src.demo.backend.main:app"
        sleep 1
        if ! is_backend_running; then
            echo -e "${GREEN}FastAPI backend stopped successfully.${NC}"
        else
            echo -e "${RED}Failed to stop FastAPI backend.${NC}"
        fi
    else
        echo -e "${YELLOW}FastAPI backend is not running.${NC}"
    fi
}

# Stop frontend
stop_frontend() {
    if is_frontend_running; then
        echo -e "${GREEN}Stopping frontend...${NC}"
        pkill -f "streamlit run src/demo/frontend/app.py"
        sleep 1
        if ! is_frontend_running; then
            echo -e "${GREEN}Frontend stopped successfully.${NC}"
        else
            echo -e "${RED}Failed to stop frontend.${NC}"
        fi
    else
        echo -e "${YELLOW}Frontend is not running.${NC}"
    fi
}

# Main command handler
case "$1" in
    start)
        start_backend
        start_frontend
        ;;
    stop)
        stop_frontend
        stop_backend
        ;;
    *)
        echo -e "${RED}Usage: $0 {start|stop}${NC}"
        exit 1
        ;;
esac