# Frontend auf anderem Gerät ausführen

Diese Anleitung erklärt, wie du das Frontend auf einem anderen Gerät ausführst, während das Backend und Ollama auf deinem PC laufen.

## Architektur

```
Gerät 1 (Dein PC):
├── Ollama (Port 11434) - lokal
├── Backend (Port 8000) - öffentlich erreichbar
└── Frontend (optional, Port 3000)

Gerät 2 (Anderes Gerät):
└── Frontend (Port 3000) - zeigt auf Backend auf Gerät 1
```

## Schritt 1: Backend auf deinem PC starten

Auf deinem PC (wo Ollama läuft):

```bash
cd ~/Desktop/RagAnything/webapp_api
python3 full_rag_server.py
```

Der Server läuft auf `0.0.0.0:8000` und ist öffentlich erreichbar.

## Schritt 2: IP-Adresse deines PCs herausfinden

Auf deinem PC:

```bash
# macOS
ipconfig getifaddr en0
# Oder
ifconfig | grep "inet " | grep -v 127.0.0.1

# Linux
hostname -I

# Windows
ipconfig | findstr IPv4
```

**Beispiel:** `100.124.29.158`

## Schritt 3: Frontend auf anderem Gerät konfigurieren

### Option A: Development Server (für Entwicklung)

1. **Code auf anderes Gerät kopieren:**
   ```bash
   # Auf anderem Gerät
   scp -r ~/Desktop/RagAnything/webapp user@anderes-geraet:~/rag-frontend
   ```

2. **Umgebungsvariable setzen:**
   ```bash
   cd ~/rag-frontend/webapp
   
   # Erstelle .env Datei
   echo "REACT_APP_API_URL=http://100.124.29.158:8000" > .env
   
   # Oder manuell erstellen:
   # REACT_APP_API_URL=http://DEINE-PC-IP:8000
   ```

3. **Dependencies installieren:**
   ```bash
   npm install
   ```

4. **Frontend starten:**
   ```bash
   npm start
   ```

### Option B: Production Build (für Produktion)

1. **Build auf deinem PC erstellen:**
   ```bash
   cd ~/Desktop/RagAnything/webapp
   
   # Setze Backend-URL für Build
   echo "REACT_APP_API_URL=http://100.124.29.158:8000" > .env
   
   # Build erstellen
   npm run build
   ```

2. **Build auf anderes Gerät kopieren:**
   ```bash
   # Auf deinem PC
   scp -r ~/Desktop/RagAnything/webapp/build user@anderes-geraet:~/rag-frontend
   ```

3. **Auf anderem Gerät - Build-Server starten:**
   ```bash
   # Auf anderem Gerät
   cd ~/rag-frontend
   
   # Serve installieren (falls nicht vorhanden)
   npm install -g serve
   
   # Build-Server starten
   serve -s build -l 3000
   ```

## Schritt 4: Netzwerk-Konfiguration

### Gleiches Netzwerk (LAN/WLAN)

Wenn beide Geräte im gleichen Netzwerk sind:
- Backend-URL: `http://<deine-pc-ip>:8000`
- Beispiel: `http://100.124.29.158:8000`

### Verschiedene Netzwerke

Wenn die Geräte in verschiedenen Netzwerken sind:

1. **Router Port-Forwarding einrichten:**
   - Port 8000 → deine PC IP:8000
   - Öffentliche IP herausfinden: `curl ifconfig.me`

2. **Backend-URL anpassen:**
   ```bash
   REACT_APP_API_URL=http://<deine-öffentliche-ip>:8000
   ```

3. **Oder ngrok verwenden:**
   ```bash
   # Auf deinem PC
   ngrok http 8000
   # Verwende die ngrok-URL als Backend-URL
   ```

## Schritt 5: Firewall konfigurieren

### Auf deinem PC (wo Backend läuft):

**macOS:**
```bash
# Firewall-Einstellungen öffnen
open /System/Library/PreferencePanes/Security.prefPane

# Port 8000 freigeben
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /usr/local/bin/python3
```

**Linux:**
```bash
sudo ufw allow 8000/tcp
sudo ufw reload
```

**Windows:**
- Windows Defender Firewall → Erweiterte Einstellungen
- Eingehende Regel → Port 8000 erlauben

## Schritt 6: Testen

### Auf anderem Gerät:

1. **Backend-Verbindung testen:**
   ```bash
   curl http://100.124.29.158:8000/api/health
   ```

2. **Frontend öffnen:**
   - Development: `http://localhost:3000`
   - Production: `http://localhost:3000` (oder die IP des Geräts)

3. **Test-Query senden:**
   - Öffne die Website
   - Stelle eine Frage
   - Prüfe ob die Antwort vom Backend (dein PC) kommt

## Troubleshooting

### Frontend kann Backend nicht erreichen

1. **Prüfe Backend-URL:**
   ```bash
   # In Browser-Konsole (F12)
   console.log(process.env.REACT_APP_API_URL)
   ```

2. **Prüfe Netzwerk-Verbindung:**
   ```bash
   ping <deine-pc-ip>
   ```

3. **Prüfe Firewall:**
   - Stelle sicher, dass Port 8000 auf deinem PC freigegeben ist

### CORS-Fehler

Das Backend sollte bereits CORS konfiguriert haben. Falls nicht:

```python
# In full_rag_server.py sollte bereits stehen:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Build verwendet falsche URL

1. **Lösche Build und erstelle neu:**
   ```bash
   rm -rf build dist
   echo "REACT_APP_API_URL=http://100.124.29.158:8000" > .env
   npm run build
   ```

2. **Prüfe .env Datei:**
   ```bash
   cat .env
   ```

## Zusammenfassung

**Auf deinem PC (wo Ollama läuft):**
- ✅ Ollama starten: `ollama serve`
- ✅ Backend starten: `python3 full_rag_server.py`
- ✅ Port 8000 freigeben (Firewall)

**Auf anderem Gerät:**
- ✅ Frontend-Code kopieren
- ✅ `.env` Datei mit Backend-URL erstellen
- ✅ `npm install && npm start` (Development)
- ✅ Oder `npm run build && serve -s build` (Production)

**Backend-URL Format:**
- Gleiches Netzwerk: `http://<pc-ip>:8000`
- Verschiedene Netzwerke: `http://<öffentliche-ip>:8000` oder ngrok-URL

