#!/bin/bash
# Launch the Streamlit web UI for the P&G Deep Research project

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

source .venv/bin/activate

echo "Installing dependencies..."
pip install -q -r requirements.txt

if [ ! -f ".env" ]; then
  echo ""
  echo "WARNING: .env file not found."
  echo "Copy .env.example to .env and add your TAVILY_API_KEY"
  echo ""
fi

echo ""
echo "Starting web UI at http://localhost:8501"
echo "Press Ctrl+C to stop"
echo ""

streamlit run app.py --server.address localhost --server.port 8501
