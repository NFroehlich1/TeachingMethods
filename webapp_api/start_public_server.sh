#!/bin/bash

# Prüfen ob npx verfügbar ist
if ! command -v npx &> /dev/null; then
    echo "npx command could not be found. Please install Node.js."
    exit 1
fi

echo "========================================================"
echo "Starte RAG Public Tunnel mit localtunnel..."
echo "========================================================"
echo "Dieser Tunnel macht deine lokale Instanz (http://localhost:8000)"
echo "öffentlich im Internet verfügbar."
echo ""
echo "HINWEIS: Du musst deine lokale Server-App separat laufen lassen!"
echo ""

# Start localtunnel
npx localtunnel --port 8000
