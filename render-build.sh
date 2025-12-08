#!/usr/bin/env bash
# exit on error
set -o errexit

# Install Python dependencies
pip install -r requirements.txt

# Build React Frontend
echo "Building React Frontend..."
cd webapp
npm install
npm run build
cd ..

# Move dist to expected location if needed (optional check)
if [ -d "webapp/dist" ]; then
    echo "Frontend build successful. Dist folder created."
else
    echo "Error: Dist folder not found!"
    exit 1
fi

