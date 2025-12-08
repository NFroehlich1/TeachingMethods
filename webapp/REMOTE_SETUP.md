# Frontend auf anderem Gerät ausführen - Schnellstart

## Auf deinem PC (wo Ollama läuft):

1. **Backend starten:**
   ```bash
   cd ~/Desktop/RagAnything/webapp_api
   python3 full_rag_server.py
   ```

2. **IP-Adresse herausfinden:**
   ```bash
   ipconfig getifaddr en0
   # Beispiel: 100.124.29.158
   ```

## Auf anderem Gerät:

### Option 1: Development Server

```bash
# 1. Code kopieren (oder git clone)
cd ~/rag-frontend/webapp

# 2. .env Datei erstellen
echo "REACT_APP_API_URL=http://100.124.29.158:8000" > .env
# Ersetze 100.124.29.158 mit der IP deines PCs!

# 3. Installieren und starten
npm install
npm start
```

### Option 2: Production Build

```bash
# 1. Build auf deinem PC erstellen (mit korrekter Backend-URL)
cd ~/Desktop/RagAnything/webapp
echo "REACT_APP_API_URL=http://100.124.29.158:8000" > .env
npm run build

# 2. Build auf anderes Gerät kopieren
scp -r build user@anderes-geraet:~/rag-frontend

# 3. Auf anderem Gerät - Server starten
cd ~/rag-frontend
npx serve -s build -l 3000
```

## Wichtig:

- ✅ Backend muss auf deinem PC laufen (Port 8000)
- ✅ Ollama muss auf deinem PC laufen (Port 11434)
- ✅ Firewall muss Port 8000 freigeben
- ✅ Beide Geräte müssen im gleichen Netzwerk sein (oder Port-Forwarding)

