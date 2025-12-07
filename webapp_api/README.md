# RAG Chatbot API Server

FastAPI Backend für die React-Weboberfläche.

## Installation

```bash
pip install -r requirements.txt
```

## Starten

```bash
python server.py
```

Der Server läuft dann auf http://localhost:8000

## API Endpoints

### GET /
Health check

### GET /api/health
API Status

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
  "message": "Document 'example.pdf' processed successfully",
  "filename": "example.pdf"
}
```

### GET /api/documents
Listet alle verarbeiteten Dokumente auf.

## Umgebungsvariablen

Der Server benötigt die folgenden Umgebungsvariablen (aus `.env`):

- `SUPABASE_KEY` - Supabase Anon Key
- `SUPABASE_ACCESS_TOKEN` - Supabase Access Token
- `SUPABASE_DB_PASSWORD` - Supabase Database Password
- `GEMINI_API_KEY` - Gemini API Key

## Entwicklung

Für Entwicklung mit Auto-Reload:

```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

