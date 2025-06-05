import asyncio
import websockets
import json
from dotenv import load_dotenv
import os
from google import genai
from google.genai import types
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
PLACES_API_KEY = os.getenv("PLACES_API_KEY")
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "http://localhost:5678/webhook/chatbot")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is missing!")

client = genai.Client(api_key=API_KEY)
model_id = "gemini-2.0-flash-live-001"

async def handle_websocket(websocket):

    SYSTEM_PROMPT = """
You are a versatile and friendly AI assistant."""
    
    try:
        config = types.LiveConnectConfig(
            response_modalities=[types.Modality.TEXT],
            system_instruction=types.Content(parts=[types.Part(text=SYSTEM_PROMPT)]),
        )

        async with client.aio.live.connect(model=model_id, config=config) as session:
            # TASK 1: Receive from Gemini and send to Client
            async def gemini_to_client():
                try:
                    async for gemini_message in session.receive():
                        if gemini_message.text:
                            await websocket.send(json.dumps({
                                "type": "chunk", "response_chunk": gemini_message.text
                            }))
                            print("Gemini sent text:", gemini_message.text)

                        if gemini_message.server_content and gemini_message.server_content.turn_complete:
                            await websocket.send(json.dumps({"type": "final"}))
                            print("Gemini turn complete.")

                except Exception as e:
                    print(f"Error in gemini_to_client task: {e}")
                    await websocket.send(json.dumps({"error": str(e)}))

            # TASK 2: Receive from Client and send to Gemini
            async def client_to_gemini():
                try:
                    async for message in websocket:
                        data = json.loads(message)
                        user_input = data.get("text")
                        if user_input:
                            print("Client sent text:", user_input)
                            await session.send_client_content(turns={"role": "user", "parts": [{"text": user_input}]}, turn_complete=True)
                            print("Client sent text to Gemini.")
                except Exception as e:
                    print(f"Error in client_to_gemini task: {e}")
                    await websocket.send(json.dumps({"error": str(e)}))

            await asyncio.gather(gemini_to_client(), client_to_gemini())
    
            print("Session ended.")

    except Exception as e:
        print(f"Overall websocket error: {e}")
        try:
            await websocket.send(json.dumps({"error": str(e)}))
        except websockets.exceptions.ConnectionClosed:
            pass

async def main():
    async with websockets.serve(handle_websocket, "0.0.0.0", 8765):
        print("WebSocket server started on ws://0.0.0.0:8765")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())