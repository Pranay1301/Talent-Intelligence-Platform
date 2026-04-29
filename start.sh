#!/bin/bash

# Start FastAPI backend in the background
echo "Starting FastAPI backend..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Wait a few seconds for backend to start
sleep 3

# Start Streamlit dashboard in the foreground on the port Render provides
echo "Starting Streamlit dashboard..."
streamlit run dashboard/streamlit_app.py --server.port ${PORT:-8501} --server.address 0.0.0.0
