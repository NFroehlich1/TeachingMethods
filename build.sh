#!/bin/bash
set -e

echo "Building React frontend..."
cd webapp
npm install
npm run build
cd ..

echo "Frontend build complete. Starting server..."
python webapp_api/full_rag_server.py

