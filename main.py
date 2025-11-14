from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from typing import List, Optional
from dotenv import load_dotenv
import asyncio
import websockets
from groq import Groq
import re
import httpx
import uvicorn
# Hugging Face imports commented out - using Groq instead
# from huggingface_hub import InferenceClient
# from huggingface_hub.errors import HfHubHTTPError
# Load environment variables from .env or env file
# Try .env first (standard), then fall back to 'env' file
if os.path.exists(".env"):
    load_dotenv(".env")
elif os.path.exists("env"):
    load_dotenv("env")
else:
    load_dotenv()  # Try default .env location

app = FastAPI(title="Sentiment Aura API", version="1.0.0")

# CORS middleware to allow frontend connections
# IMPORTANT: Must be added before routes are defined
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Must be False when using ["*"]
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],  # Expose all headers
)

# Request/Response models
class ProcessTextRequest(BaseModel):
    text: str

class ProcessTextResponse(BaseModel):
    sentiment: float  # -1 to 1
    keywords: List[str]

class DeepgramURLResponse(BaseModel):
    url: str
    token: Optional[str] = None

# Initialize Groq client for sentiment analysis
groq_client = None
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
    print("Groq client initialized successfully")
else:
    print("Warning: GROQ_API_KEY not found in environment variables")

# Hugging Face code commented out - using Groq instead
# # Initialize Hugging Face Inference client
# hf_client = None
# hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
# if hf_api_key:
#     # Initialize with API key - newer versions of huggingface-hub (>=0.28.0)
#     # should automatically use the new router endpoint
#     hf_client = InferenceClient(token=hf_api_key)
# 
# # Sentiment analysis model
# SENTIMENT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

@app.get("/")
async def root():
    return {"message": "Sentiment Aura API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    """Handle OPTIONS requests for CORS preflight"""
    return {"message": "OK"}

@app.get("/api/cors-test")
async def cors_test():
    """Test endpoint to verify CORS is working"""
    return {
        "message": "CORS is working!",
        "origin_allowed": True,
        "cors_configured": True
    }

@app.get("/api/debug/deepgram-key")
async def debug_deepgram_key():
    """
    Debug endpoint to check if Deepgram API key is loaded (first 10 chars only for security)
    """
    deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
    if deepgram_api_key:
        # Test if we can make a simple request to verify the key
        test_url = "https://api.deepgram.com/v1/projects"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    test_url,
                    headers={"Authorization": f"Token {deepgram_api_key}"},
                    timeout=5.0
                )
                key_valid = response.status_code == 200
                return {
                    "loaded": True,
                    "key_length": len(deepgram_api_key),
                    "key_preview": deepgram_api_key[:10] + "..." if len(deepgram_api_key) > 10 else "too short",
                    "key_valid": key_valid,
                    "test_status": response.status_code if not key_valid else None,
                    "url_preview": f"wss://api.deepgram.com/v1/listen?token={'*' * 10}&model=nova-2&..."
                }
        except Exception as e:
            return {
                "loaded": True,
                "key_length": len(deepgram_api_key),
                "key_preview": deepgram_api_key[:10] + "..." if len(deepgram_api_key) > 10 else "too short",
                "key_valid": "unknown",
                "test_error": str(e),
                "url_preview": f"wss://api.deepgram.com/v1/listen?token={'*' * 10}&model=nova-2&..."
            }
    return {"loaded": False, "error": "DEEPGRAM_API_KEY not found in environment"}

@app.websocket("/ws/deepgram")
async def websocket_proxy(websocket: WebSocket):
    """
    WebSocket proxy that connects to Deepgram with proper authentication.
    Frontend connects to this endpoint, and we proxy to Deepgram with Authorization header.
    """
    await websocket.accept()
    
    deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
    if not deepgram_api_key:
        try:
            await websocket.close(code=1008, reason="DEEPGRAM_API_KEY not configured")
        except:
            pass
        return
    
    # Connect to Deepgram using Sec-WebSocket-Protocol for authentication
    # This is the browser-compatible way to pass authentication
    deepgram_url = "wss://api.deepgram.com/v1/listen?model=nova-2&language=en-US&punctuate=true&interim_results=true"
    
    try:
        # Use websockets library with subprotocols for authentication
        # Deepgram accepts token via Sec-WebSocket-Protocol header
        # Format: ['token', 'YOUR_API_KEY'] as separate subprotocols
        async with websockets.connect(
            deepgram_url,
            subprotocols=['token', deepgram_api_key]
        ) as deepgram_ws:
            # Create tasks to forward messages in both directions
            async def forward_to_deepgram():
                try:
                    while True:
                        # Receive binary audio data from frontend
                        data = await websocket.receive_bytes()
                        # Forward binary data to Deepgram
                        await deepgram_ws.send(data)
                except WebSocketDisconnect:
                    print("Frontend disconnected")
                except websockets.exceptions.ConnectionClosed:
                    print("Deepgram connection closed")
                except Exception as e:
                    print(f"Error forwarding to Deepgram: {e}")
            
            async def forward_from_deepgram():
                try:
                    while True:
                        message = await deepgram_ws.recv()
                        # Deepgram sends JSON strings
                        try:
                            if isinstance(message, str):
                                # Log the message for debugging
                                import json
                                try:
                                    msg_data = json.loads(message)
                                    msg_type = msg_data.get("type", "unknown")
                                    transcript = ""
                                    is_final = False
                                    
                                    if msg_type == "Results":
                                        transcript = msg_data.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "")
                                        is_final = msg_data.get("is_final", False)
                                    elif msg_type == "TurnInfo":
                                        transcript = msg_data.get("transcript", "")
                                        is_final = msg_data.get("event") == "EndOfTurn"
                                    
                                    if transcript:
                                        print(f"Deepgram message: type={msg_type}, is_final={is_final}, transcript='{transcript[:50]}...'")
                                    else:
                                        print(f"Deepgram message: type={msg_type}, event={msg_data.get('event', 'N/A')}")
                                except:
                                    print(f"Deepgram message (raw): {message[:200]}")
                                
                                await websocket.send_text(message)
                            else:
                                await websocket.send_bytes(message)
                        except (WebSocketDisconnect, RuntimeError):
                            # Frontend disconnected or connection already closed
                            break
                        except Exception as send_error:
                            print(f"Error sending to frontend: {send_error}")
                            break
                except websockets.exceptions.ConnectionClosed:
                    print("Deepgram connection closed")
                except Exception as e:
                    print(f"Error forwarding from Deepgram: {e}")
            
            # Run both forwarding tasks concurrently
            try:
                await asyncio.gather(
                    forward_to_deepgram(),
                    forward_from_deepgram(),
                    return_exceptions=True
                )
            except Exception as e:
                print(f"Error in forwarding tasks: {e}")
    except Exception as e:
        print(f"Deepgram WebSocket error: {e}")
        try:
            await websocket.close(code=1011, reason=f"Deepgram connection failed: {str(e)[:100]}")
        except:
            pass

@app.post("/api/get-deepgram-url", response_model=DeepgramURLResponse)
async def get_deepgram_url():
    """
    Returns the WebSocket URL for our proxy endpoint.
    Frontend should connect to our backend WebSocket endpoint, which proxies to Deepgram.
    """
    # Get the backend URL from environment or use localhost for development
    backend_url = os.getenv("RENDER_EXTERNAL_URL") or os.getenv("BACKEND_URL") or "http://localhost:4000"
    
    # Convert HTTP/HTTPS to WebSocket URL
    if backend_url.startswith("https://"):
        ws_url = backend_url.replace("https://", "wss://")
    elif backend_url.startswith("http://"):
        ws_url = backend_url.replace("http://", "ws://")
    else:
        # If no protocol, assume HTTPS in production
        if "localhost" in backend_url or "127.0.0.1" in backend_url:
            ws_url = "ws://" + backend_url
        else:
            ws_url = "wss://" + backend_url
    
    proxy_url = f"{ws_url}/ws/deepgram"
    
    return DeepgramURLResponse(url=proxy_url, token=None)

def extract_keywords(text: str, max_keywords: int = 8) -> List[str]:
    """
    Extract keywords from text using simple NLP techniques.
    Removes common stop words and extracts meaningful words.
    """
    # Simple stop words list
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that',
        'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me',
        'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our',
        'their', 'what', 'which', 'who', 'whom', 'whose', 'where', 'when',
        'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
        'so', 'than', 'too', 'very', 'just', 'now'
    }
    
    # Convert to lowercase and split into words
    words = re.findall(r'\b[a-z]+\b', text.lower())
    
    # Filter out stop words and short words (less than 3 characters)
    keywords = [word for word in words if word not in stop_words and len(word) >= 3]
    
    # Count word frequencies
    word_freq = {}
    for word in keywords:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_keywords[:max_keywords]]

@app.post("/process_text", response_model=ProcessTextResponse)
async def process_text(request: ProcessTextRequest):
    """
    Processes text to extract sentiment and keywords using Groq API.
    Returns sentiment score (-1 to 1) and list of keywords.
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    if not groq_client:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY not configured. Please set it in your environment variables."
        )
    
    # Retry logic with exponential backoff for timeout errors
    max_retries = 3
    retry_delays = [2, 4, 8]  # seconds to wait between retries
    
    sentiment = 0.0
    keywords = []  # Initialize keywords list
    last_error = None
    
    for attempt in range(max_retries):
        try:
            if attempt == 0:
                print(f"Using Groq API for sentiment analysis and keyword extraction")
                print(f"Groq Client initialized: {groq_client is not None}")
            
            # Create a prompt for Groq to analyze sentiment and extract keywords
            prompt = f"""Analyze the sentiment of the following text and extract the most important keywords.

Text: "{request.text}"

Provide:
1. A sentiment score between -1 and 1, where:
   - -1.0 to -0.6 = Very Negative
   - -0.6 to -0.2 = Negative  
   - -0.2 to 0.2 = Neutral
   - 0.2 to 0.6 = Positive
   - 0.6 to 1.0 = Very Positive

2. A list of 5-8 most important keywords that capture the main topics, themes, or concepts in the text. Focus on meaningful, substantive words that represent key ideas, not common words like "the", "and", "is", etc.

Respond with ONLY a JSON object in this exact format:
{{"sentiment": <number between -1 and 1>, "keywords": ["keyword1", "keyword2", "keyword3", ...]}}

Do not include any other text, only the JSON object. The keywords array should contain 5-8 strings."""

            # Call Groq API
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a sentiment analysis and keyword extraction expert. Always respond with valid JSON only. Extract meaningful, substantive keywords that represent key topics and themes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=150  # Increased for keywords
            )
            
            # Parse the response
            response_text = response.choices[0].message.content.strip()
            print(f"Groq response: {response_text}")
            
            # Extract JSON from response (handle cases where there might be extra text)
            import json
            
            # Try to find JSON in the response
            try:
                # Remove markdown code blocks if present
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0].strip()
                
                result = json.loads(response_text)
                sentiment = float(result.get("sentiment", 0.0))
                
                # Extract keywords from response
                keywords_raw = result.get("keywords", [])
                if isinstance(keywords_raw, list):
                    # Clean and validate keywords
                    keywords = [
                        str(k).strip().lower() 
                        for k in keywords_raw 
                        if k and len(str(k).strip()) >= 3
                    ][:8]  # Limit to 8 keywords
                else:
                    # Fallback: try to extract keywords from string
                    keywords = []
                
                # Clamp sentiment to -1 to 1 range
                sentiment = max(-1.0, min(1.0, sentiment))
                
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract number from response
                import re
                numbers = re.findall(r'-?\d+\.?\d*', response_text)
                if numbers:
                    sentiment = float(numbers[0])
                    sentiment = max(-1.0, min(1.0, sentiment))
                else:
                    raise ValueError(f"Could not parse sentiment from response: {response_text}")
                # If JSON parsing fails, use local NLP as fallback
                keywords = extract_keywords(request.text)
            
            break  # Success, exit retry loop
            
        except Exception as e:
            last_error = e
            error_msg = str(e)
            
            # Check if it's a timeout or gateway error (retryable)
            is_retryable = (
                "504" in error_msg or 
                "Gateway Time-out" in error_msg or
                "timeout" in error_msg.lower() or
                "503" in error_msg or
                "502" in error_msg or
                "500" in error_msg or
                "rate limit" in error_msg.lower()
            )
            
            if is_retryable and attempt < max_retries - 1:
                wait_time = retry_delays[attempt]
                print(f"Attempt {attempt + 1} failed with {type(e).__name__}: {error_msg[:100]}")
                print(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
                continue
            else:
                # Not retryable or out of retries
                print(f"Error in process_text (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {error_msg}")
                if attempt == max_retries - 1:
                    raise HTTPException(
                        status_code=503,
                        detail=f"Groq API error after {max_retries} attempts: {error_msg}"
                    )
                raise
    
    try:
        # Ensure sentiment is in range [-1, 1]
        sentiment = max(-1.0, min(1.0, sentiment))
        
        # If keywords weren't extracted from Groq (fallback case), use local NLP
        if not keywords:
            print("Warning: No keywords from Groq, using local NLP fallback")
            keywords = extract_keywords(request.text)
        
        # Ensure we have keywords (at least empty list)
        if not keywords:
            keywords = []
        
        return ProcessTextResponse(sentiment=sentiment, keywords=keywords)
        
    except HTTPException:
        # Re-raise HTTPException (from retry logic) without modification
        raise
    except Exception as e:
        # Log the full error for debugging
        error_msg = str(e)
        error_type = type(e).__name__
        print(f"Error in process_text: {error_type}: {error_msg}")
        import traceback
        traceback.print_exc()
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing text: {error_msg}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000)

