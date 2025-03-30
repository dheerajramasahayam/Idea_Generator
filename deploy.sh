#!/bin/bash

# Simple deployment script for the SaaS Idea Automator
# Assumes running on the server in the project's parent directory or using full paths

# --- Configuration ---
PROJECT_DIR="/opt/saas-automator" # !!! IMPORTANT: Adjust if your project path is different
GIT_BRANCH="main" # Or the branch you use for deployment
SERVICE_NAME="saas-automator.service"
# --- End Configuration ---

set -e # Exit immediately if a command exits with a non-zero status.

echo "=== Starting Deployment ==="

echo "[1/5] Navigating to project directory: ${PROJECT_DIR}"
cd "${PROJECT_DIR}" || { echo "Failed to cd into ${PROJECT_DIR}"; exit 1; }

echo "[2/5] Pulling latest changes from Git (branch: ${GIT_BRANCH})..."
git checkout ${GIT_BRANCH} || { echo "Failed to checkout branch ${GIT_BRANCH}"; exit 1; }
git pull origin ${GIT_BRANCH} || { echo "Failed to pull changes"; exit 1; }

echo "[3/5] Activating virtual environment..."
source venv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }

echo "[4/5] Installing/updating dependencies..."
pip install -r requirements.txt || { echo "Failed to install dependencies"; exit 1; }
# Optional: Add NLTK data download if using NLTK for trend analysis later
# python -m nltk.downloader punkt stopwords # Example

echo "[5/5] Restarting systemd service: ${SERVICE_NAME}..."
sudo systemctl restart ${SERVICE_NAME} || { echo "Failed to restart service ${SERVICE_NAME}"; exit 1; }

echo "--- Deployment finished successfully! ---"

echo "Checking service status (wait a few seconds)..."
sleep 3
sudo systemctl status ${SERVICE_NAME} --no-pager # --no-pager prevents needing to press 'q'

echo "You may want to check the logs:"
echo "sudo journalctl -u ${SERVICE_NAME} -f"
echo "tail -f ${PROJECT_DIR}/automator.log"
echo "tail -f ${PROJECT_DIR}/automator_stderr.log"
