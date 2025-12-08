# Ollama-Modell auf deinem PC hosten

Diese Anleitung erklärt, wie du dein Ollama-Modell so konfigurierst, dass es über deinen PC gehostet wird und von außen erreichbar ist.

## Aktuelle Konfiguration

**Wie es funktioniert:**
1. **Ollama** läuft lokal auf deinem PC (`localhost:11434`)
2. **Backend-Server** läuft auf deinem PC (`0.0.0.0:8000` - öffentlich erreichbar)
3. **Frontend** läuft auf deinem PC (`0.0.0.0:3000` - öffentlich erreichbar)
4. Wenn jemand von außen eine Anfrage stellt:
   - Frontend → Backend (auf deinem PC)
   - Backend → Ollama (lokal auf deinem PC)
   - Ollama verarbeitet die Anfrage
   - Antwort geht zurück: Ollama → Backend → Frontend → Benutzer

**Vorteil:** Ollama muss NICHT öffentlich erreichbar sein! Das Backend leitet alle Anfragen weiter.

## Schritt 1: Ollama starten

### Option A: Ollama als Service starten (empfohlen)
```bash
# Ollama im Hintergrund starten
ollama serve

# Oder mit spezifischem Modell
ollama run gpt-oss:20b
```

### Option B: Ollama als System-Service (macOS)
```bash
# Erstelle LaunchAgent für automatischen Start
mkdir -p ~/Library/LaunchAgents

cat > ~/Library/LaunchAgents/com.ollama.server.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.ollama.server</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/ollama</string>
        <string>serve</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
EOF

# Service laden
launchctl load ~/Library/LaunchAgents/com.ollama.server.plist
```

## Schritt 2: Backend-Server starten

```bash
cd ~/Desktop/RagAnything/webapp_api
python3 full_rag_server.py
```

Der Server läuft auf `0.0.0.0:8000` und ist öffentlich erreichbar.

## Schritt 3: Konfiguration prüfen

### Prüfe ob Ollama läuft:
```bash
curl http://localhost:11434/api/tags
```

### Prüfe ob Backend läuft:
```bash
curl http://localhost:8000/api/health
```

### Teste die Verbindung:
```bash
# Teste ob Backend Ollama erreichen kann
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello, test", "llm_provider": "ollama"}'
```

## Schritt 4: Öffentlichen Zugriff einrichten

### Option A: Router Port-Forwarding (für lokales Netzwerk)

1. **Finde deine lokale IP:**
   ```bash
   ipconfig getifaddr en0  # macOS
   # Oder: ifconfig | grep "inet "
   ```

2. **Router-Admin öffnen** (meist `192.168.1.1` oder `192.168.0.1`)

3. **Port-Forwarding einrichten:**
   - Port 8000 → deine lokale IP:8000 (Backend)
   - Port 3000 → deine lokale IP:3000 (Frontend)
   - Port 11434 → NICHT nötig! (Ollama bleibt lokal)

4. **Öffentliche IP herausfinden:**
   ```bash
   curl ifconfig.me
   ```

5. **Zugriff:**
   - Backend: `http://<deine-öffentliche-ip>:8000`
   - Frontend: `http://<deine-öffentliche-ip>:3000`

### Option B: ngrok (einfach, für Tests)

```bash
# ngrok installieren: https://ngrok.com/download

# Backend tunnel
ngrok http 8000

# Frontend tunnel (in neuem Terminal)
ngrok http 3000
```

### Option C: Cloudflare Tunnel (kostenlos, dauerhaft)

```bash
# Cloudflare Tunnel installieren
# https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/

# Tunnel erstellen
cloudflared tunnel create rag-server
cloudflared tunnel route dns rag-server rag.yourdomain.com
cloudflared tunnel run rag-server
```

## Schritt 5: Firewall konfigurieren

### macOS Firewall:
```bash
# Firewall-Einstellungen öffnen
open /System/Library/PreferencePanes/Security.prefPane

# Oder über Terminal:
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate on
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /usr/local/bin/python3
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /usr/local/bin/ollama
```

### Ports freigeben:
- Port 8000: Backend API
- Port 3000: Frontend
- Port 11434: NICHT nötig (Ollama bleibt lokal)

## Schritt 6: Performance-Optimierung

### Ollama GPU-Unterstützung (falls verfügbar):
```bash
# Prüfe ob GPU verfügbar ist
ollama show gpt-oss:20b --modelfile

# GPU-Nutzung aktivieren (automatisch, wenn verfügbar)
export OLLAMA_NUM_GPU=1
```

### Modell-Größe anpassen:
```bash
# Kleinere Modelle für bessere Performance
ollama pull llama2:7b  # Statt 20b
```

### Server-Ressourcen:
- **RAM:** Mindestens 8GB (16GB+ empfohlen für 20B Modelle)
- **CPU:** Mehr Kerne = bessere Performance
- **GPU:** Deutlich bessere Performance (falls verfügbar)

## Troubleshooting

### Ollama nicht erreichbar:
```bash
# Prüfe ob Ollama läuft
ps aux | grep ollama

# Starte Ollama neu
pkill ollama
ollama serve
```

### Backend kann Ollama nicht erreichen:
```bash
# Prüfe Backend-Logs
tail -f /tmp/rag_server.log

# Teste Ollama direkt
curl http://localhost:11434/api/generate \
  -d '{"model": "gpt-oss:20b", "prompt": "test"}'
```

### Port bereits belegt:
```bash
# Finde Prozess auf Port 8000
lsof -i :8000

# Beende Prozess
kill -9 <PID>
```

### Externe Verbindungen funktionieren nicht:
1. Prüfe Firewall-Einstellungen
2. Prüfe Router Port-Forwarding
3. Prüfe ob Server auf `0.0.0.0` läuft (nicht nur `127.0.0.1`)

## Sicherheit

⚠️ **WICHTIG:** Öffentlicher Server ohne Authentifizierung!

1. **Rate Limiting:** Implementiere Rate Limits für API-Endpunkte
2. **HTTPS:** Verwende HTTPS (z.B. mit Cloudflare Tunnel)
3. **Authentifizierung:** Füge API-Keys oder Login hinzu
4. **Firewall:** Nur notwendige Ports öffnen

## Monitoring

```bash
# Ollama-Status
curl http://localhost:11434/api/tags

# Backend-Status
curl http://localhost:8000/api/health

# Server-Logs
tail -f /tmp/rag_server.log

# System-Ressourcen
top
# Oder: htop (falls installiert)
```

## Zusammenfassung

**Aktuelle Architektur:**
```
Externer Benutzer
    ↓
Frontend (Port 3000) - öffentlich
    ↓
Backend (Port 8000) - öffentlich
    ↓
Ollama (Port 11434) - NUR lokal
    ↓
Antwort zurück
```

**Vorteile:**
- ✅ Ollama bleibt sicher lokal
- ✅ Keine direkte Verbindung von außen zu Ollama nötig
- ✅ Backend kontrolliert alle Anfragen
- ✅ Einfache Konfiguration

**Nachteile:**
- ⚠️ Dein PC muss laufen, damit der Service verfügbar ist
- ⚠️ Performance hängt von deinem PC ab

