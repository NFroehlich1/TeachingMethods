#!/bin/bash
# Script zum Setup des Frontends auf einem anderen Gerät

echo "============================================================"
echo "Frontend Setup für Remote-Gerät"
echo "============================================================"
echo ""

# Frage nach Backend-URL
read -p "Backend-URL (z.B. http://100.124.29.158:8000): " BACKEND_URL

if [ -z "$BACKEND_URL" ]; then
    echo "⚠️  Keine URL eingegeben. Verwende localhost:8000"
    BACKEND_URL="http://localhost:8000"
fi

echo ""
echo "Erstelle .env Datei mit Backend-URL: $BACKEND_URL"
echo "REACT_APP_API_URL=$BACKEND_URL" > .env

echo ""
echo "✓ .env Datei erstellt"
echo ""
echo "Nächste Schritte:"
echo "1. npm install"
echo "2. npm start (für Development)"
echo "   ODER"
echo "2. npm run build && npx serve -s build -l 3000 (für Production)"
echo ""

