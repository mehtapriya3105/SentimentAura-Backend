"""
Test script to verify Deepgram WebSocket connection
"""
import os
import asyncio
import websockets
from dotenv import load_dotenv

load_dotenv("env")

async def test_deepgram_websocket():
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        print("ERROR: DEEPGRAM_API_KEY not found")
        return
    
    print(f"Testing with API key: {api_key[:10]}...")
    url1 = f"wss://api.deepgram.com/v1/listen?token={api_key}&model=nova-2&language=en-US&punctuate=true&interim_results=true"
    url2 = f"wss://api.deepgram.com/v1/listen?model=nova-2&language=en-US&punctuate=true&interim_results=true"
    url = url1
    print(f"Connecting to: wss://api.deepgram.com/v1/listen?token=***&model=nova-2&...")
    try:
        headers = {"Authorization": f"Token {api_key}"}
        print("Trying with Authorization header...")
        async with websockets.connect(url2, extra_headers=headers) as ws:
            print("✓ WebSocket connection established!")
            test_message = {
                "type": "Configure",
                "channels": 1,
                "sample_rate": 16000,
                "encoding": "linear16"
            }
            await ws.send_json(test_message)
            print("✓ Configuration message sent")
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                print(f"✓ Received response: {response[:100]}...")
            except asyncio.TimeoutError:
                print("⚠ No response received (this might be normal)")
            
            print("✓ Connection test successful!")
            
    except websockets.exceptions.InvalidStatusCode as e:
        print(f"✗ Connection failed with status code: {e.status_code}")
        print(f"  Headers: {e.headers}")
    except Exception as e:
        print(f"✗ Connection failed: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test_deepgram_websocket())

