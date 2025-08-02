#!/bin/bash
# Ensures the script exits if any command fails
set -e

# --- Title and Initial Check ---
echo "--- Local RAG Engine Launcher for Linux/macOS ---"
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null
then
    echo "ERROR: python3 is not found in your system's PATH."
    echo "Please install Python 3.8-3.12 and ensure it's available as 'python3'."
    exit 1
fi

# --- Step 1: Check for and create the virtual environment ---
echo "[1/4] Checking for virtual environment..."
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv .venv
else
    echo "Virtual environment found."
fi

# --- Step 2: Activate environment and install dependencies ---
echo ""
echo "[2/4] Activating virtual environment and installing dependencies..."
source .venv/bin/activate
pip install -r requirements.txt

# --- Step 3: Check for and download models ---
echo ""
echo "[3/4] Checking for local models..."
# Check for one of the key model folders to see if download is needed
if [ ! -d "./models/jinaai_jina-embeddings-v2-base-code" ]; then
    echo "Models not found. Running download script (this may take a while)..."
    python3 download_models.py
else
    echo "Models found locally."
fi

# --- Step 4: Launch the application ---
echo ""
echo "[4/4] Starting the RAG Application..."
echo ""
python3 app.py

echo ""
echo "Application closed."