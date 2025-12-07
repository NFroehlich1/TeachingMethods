# RAG Chatbot Web Interface

Moderne React-Weboberfläche für den RAG-Anything Chatbot mit Gemini Flash 2.0 und Supabase.

## Features

- 💬 **Chat-Interface** - Moderne Chat-Oberfläche mit Nachrichtenverlauf
- 📄 **Dokumenten-Upload** - Drag & Drop für PDF, DOC, DOCX, TXT, MD
- 🎨 **Responsive Design** - Funktioniert auf Desktop und Mobile
- ⚡ **Echtzeit-Updates** - Live-Aktualisierungen der Chat-Nachrichten
- 🔍 **Dokumenten-Verwaltung** - Übersicht aller hochgeladenen Dokumente

## Installation

```bash
cd webapp
npm install
```

## Entwicklung

```bash
npm start
```

Die App läuft dann auf http://localhost:3000

## Backend API

Die Web-App erwartet ein Backend-API auf `http://localhost:8000` mit folgenden Endpoints:

### POST /api/query
Sendet eine Anfrage an den RAG-Chatbot.

**Request:**
```json
{
  "query": "Was ist der Hauptinhalt des Dokuments?"
}
```

**Response:**
```json
{
  "response": "Die Antwort des Chatbots..."
}
```

### POST /api/upload
Lädt ein Dokument hoch und verarbeitet es.

**Request:**
- FormData mit `file` Feld

**Response:**
```json
{
  "status": "success",
  "message": "Dokument erfolgreich verarbeitet"
}
```

## Build für Produktion

```bash
npm run build
```

Die optimierte Version wird im `build/` Ordner erstellt.

## Technologien

- React 18
- Axios für API-Calls
- Lucide React für Icons
- CSS3 mit modernen Features (backdrop-filter, gradients)

## Anpassungen

### API-URL ändern

Bearbeiten Sie `src/App.js` und ändern Sie die API-URLs:

```javascript
const response = await fetch('http://localhost:8000/api/query', {
  // ...
});
```

### Styling anpassen

Die CSS-Dateien befinden sich in `src/components/` und `src/App.css`.

## Lizenz

MIT

