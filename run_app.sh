#!/bin/bash

# Real Estate Price Predictor Startup Script
echo "🏠 Starting Real Estate Price Predictor..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Run the Streamlit app
echo "🚀 Launching Streamlit app..."
streamlit run app.py --server.port 8501 --server.headless true
