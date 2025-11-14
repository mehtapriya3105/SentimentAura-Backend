# Sentiment Aura Backend

FastAPI backend for the Sentiment Aura application. Provides real-time speech transcription via Deepgram WebSocket proxy and text processing with sentiment analysis and keyword extraction using Hugging Face models.

## Features

- **Real-time Speech Transcription**: WebSocket proxy to Deepgram API for live audio transcription
- **Sentiment Analysis**: Uses Hugging Face DistilBERT model to analyze text sentiment (-1 to 1 scale)
- **Keyword Extraction**: Extracts meaningful keywords from transcribed text
- **CORS Enabled**: Configured to work with frontend applications
- **Error Handling**: Retry logic with exponential backoff for API calls

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or using a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the backend directory with the following variables:

```env
DEEPGRAM_API_KEY=your_deepgram_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
```

**Getting API Keys:**
- **Deepgram**: Sign up at https://console.deepgram.com/ and get your API key
- **Hugging Face**: Sign up at https://huggingface.co/ and create an access token at https://huggingface.co/settings/tokens

### 3. Running the Server

**Option 1: Using Python directly**
```bash
python main.py
```

**Option 2: Using uvicorn directly**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 4000
```

The API will be available at `http://localhost:4000`

## API Endpoints

### `GET /`
Health check endpoint.

**Response:**
```json
{
  "message": "Sentiment Aura API",
  "status": "running"
}
```

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

### `GET /api/debug/deepgram-key`
Debug endpoint to verify Deepgram API key configuration.

**Response:**
```json
{
  "loaded": true,
  "key_length": 40,
  "key_preview": "abc123def4...",
  "key_valid": true
}
```

### `POST /api/get-deepgram-url`
Returns the WebSocket URL for the Deepgram proxy endpoint.

**Response:**
```json
{
  "url": "ws://localhost:4000/ws/deepgram",
  "token": null
}
```

**Note:** The frontend should connect to this proxy endpoint instead of connecting directly to Deepgram. The backend handles authentication automatically.

### `POST /process_text`
Processes text to extract sentiment score and keywords.

**Request:**
```json
{
  "text": "I love this product! It's amazing and works perfectly."
}
```

**Response:**
```json
{
  "sentiment": 0.85,
  "keywords": ["love", "product", "amazing", "works", "perfectly"]
}
```

**Sentiment Scale:**
- `-1.0` to `-0.6`: Very Negative
- `-0.6` to `-0.2`: Negative
- `-0.2` to `0.2`: Neutral
- `0.2` to `0.6`: Positive
- `0.6` to `1.0`: Very Positive

**Error Responses:**
- `400`: Text is empty
- `500`: Hugging Face API key not configured
- `503`: Hugging Face API unavailable after retries

### `WebSocket /ws/deepgram`
WebSocket proxy endpoint that connects to Deepgram API with proper authentication.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:4000/ws/deepgram');
```

**Usage:**
1. Frontend connects to this endpoint
2. Send binary audio data to the WebSocket
3. Receive JSON transcription results from Deepgram
4. The backend automatically handles Deepgram authentication

**Message Format:**
- **To Server**: Binary audio data (PCM, Opus, etc.)
- **From Server**: JSON strings with transcription results

**Example Response:**
```json
{
  "channel": {
    "alternatives": [{
      "transcript": "Hello world",
      "confidence": 0.99
    }]
  }
}
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DEEPGRAM_API_KEY` | Yes | Your Deepgram API key for speech transcription |
| `HUGGINGFACE_API_KEY` | Yes | Your Hugging Face API token for sentiment analysis |

## Dependencies

- **fastapi**: Web framework for building APIs
- **uvicorn**: ASGI server for running FastAPI
- **python-dotenv**: Loading environment variables
- **websockets**: WebSocket client library for Deepgram connection
- **huggingface-hub**: Hugging Face Inference API client
- **httpx**: HTTP client for API requests
- **pydantic**: Data validation using Python type annotations

## Architecture

```
Frontend (React)
    ↓
Backend (FastAPI)
    ├── WebSocket Proxy → Deepgram API (Speech Transcription)
    └── HTTP Endpoints → Hugging Face API (Sentiment Analysis)
```

## CORS Configuration

The backend is configured to allow requests from:
- `http://localhost:8080`
- `http://localhost:5173`
- `http://127.0.0.1:8080`

To add more origins, modify the `allow_origins` list in `main.py`.

## Error Handling

- **Retry Logic**: The `/process_text` endpoint includes retry logic with exponential backoff for timeout errors
- **WebSocket Errors**: Connection errors are properly handled and logged
- **API Errors**: All endpoints return appropriate HTTP status codes and error messages

## Development

For development with auto-reload:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 4000
```

## Production

For production deployment:
```bash
uvicorn main:app --host 0.0.0.0 --port 4000 --workers 4
```

## License

This project is part of the Sentiment Aura application.
