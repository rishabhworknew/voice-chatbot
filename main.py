import asyncio
import websockets
import json
import requests
from dotenv import load_dotenv
import os
from datetime import datetime
import pytz
from tenacity import retry, stop_after_attempt, wait_fixed
from google import genai
from google.genai import types
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "http://localhost:5678/webhook/chatbot")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is missing!")

# Initialize Gemini Live API client
client = genai.Client(api_key=API_KEY)
model_id = "gemini-2.0-flash-live-001"

# System prompt for ride-booking assistant
SYSTEM_PROMPT = """
You are a ride-booking assistant in the UAE (UTC+4). Your job is to guide users conversationally to book a ride. For each user input, respond with a JSON object in the format:
{
  "response": "string",
  "state": {
    "startLocation": null or string,
    "endLocation": null or string,
    "startDate": null or string,
    "startTime": null or string,
    "selectedSlot": null or string,
    "rideConfirmation": boolean,
    "rideRejection": boolean
  }
}
- If the user provides a location, date, or time, update the corresponding state field.
- If the user asks for a discount or unrelated question, respond with "Sorry, we can't offer discounts. [Repeat previous prompt or provide relevant guidance]."
- If the user confirms or rejects the ride, set rideConfirmation or rideRejection accordingly.
- Keep responses concise and conversational.
"""

def validate_state(state):
    """Validate state fields"""
    try:
        if state["startDate"]:
            datetime.strptime(state["startDate"], "%Y-%m-%d")
        if state["startTime"]:
            datetime.strptime(state["startTime"], "%H:%M")
        return state
    except ValueError as e:
        logger.error(f"Invalid state format: {str(e)}")
        raise ValueError(f"Invalid state format: {str(e)}")

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def call_n8n_webhook(data):
    """Send structured output to n8n webhook"""
    headers = {"Content-Type": "application/json"}
    response = requests.post(N8N_WEBHOOK_URL, json=data, headers=headers)
    response.raise_for_status()
    return response.json()

async def handle_websocket(websocket):
    """Handle WebSocket connection from frontend"""
    session_id = f"{int(asyncio.get_event_loop().time())}-{uuid.uuid4().hex[:8]}"
    state = {
        "startLocation": None,
        "endLocation": None,
        "startDate": None,
        "startTime": None,
        "selectedSlot": None,
        "rideConfirmation": False,
        "rideRejection": False
    }
    uae_tz = pytz.timezone("Asia/Dubai")

    try:
        async with asyncio.timeout(300):  # 5-minute timeout for inactivity
            # Configure Live API session with system instruction
            config = types.LiveConnectConfig(
                response_modalities=[types.Modality.TEXT],
                system_instruction=types.Content(
                    parts=[types.Part(text=SYSTEM_PROMPT)]
                )
            )

            async with client.aio.live.connect(model=model_id, config=config) as session:
                async for message in websocket:
                    try:
                        # Parse incoming message
                        data = json.loads(message)
                        user_input = data.get("text")
                        audio_input = data.get("audio")  # Base64 audio (if provided)

                        if not user_input and not audio_input:
                            await websocket.send(json.dumps({
                                "error": "No input received",
                                "session_id": session_id
                            }))
                            continue

                        if not data.get("authorization"):
                            await websocket.send(json.dumps({
                                "error": "Missing authorization",
                                "session_id": session_id
                            }))
                            continue

                        # Handle audio input (convert base64 to PCM if provided)
                        if audio_input:
                            try:
                                import io
                                import soundfile as sf
                                import librosa
                                import base64

                                # Decode base64 audio
                                audio_bytes = base64.b64decode(audio_input)
                                buffer = io.BytesIO(audio_bytes)

                                # Convert to 16-bit PCM, 16kHz, mono
                                y, sr = librosa.load(buffer, sr=16000, mono=True)
                                pcm_buffer = io.BytesIO()
                                sf.write(pcm_buffer, y, sr, format='RAW', subtype='PCM_16')
                                pcm_buffer.seek(0)

                                # Send audio to Gemini
                                await session.send_realtime_input(
                                    audio=types.Blob(data=pcm_buffer.read(), mime_type="audio/pcm;rate=16000")
                                )

                                # Process Gemini response
                                response_text = []
                                async for gemini_message in session.receive():
                                    if gemini_message.text:
                                        response_text.append(gemini_message.text)
                                    if gemini_message.server_content and gemini_message.server_content.turn_complete:
                                        break

                                # Parse response
                                try:
                                    gemini_output = json.loads("".join(response_text))
                                    response = gemini_output.get("response", "Sorry, something went wrong.")
                                    state = validate_state(gemini_output.get("state", state))
                                except json.JSONDecodeError:
                                    response = "Sorry, something went wrong. Please try again."
                                    state = state
                            except Exception as e:
                                await websocket.send(json.dumps({
                                    "error": f"Audio processing error: {str(e)}",
                                    "session_id": session_id
                                }))
                                continue
                        else:
                            # Handle text input
                            try:
                                # Use send_client_content instead of send
                                await session.send_client_content(
                                    turns={
                                        "role": "user",
                                        "parts": [{"text": user_input}]
                                    },
                                    turn_complete=True
                                )

                                # Process Gemini response
                                response_text = []
                                async for gemini_message in session.receive():
                                    if gemini_message.text:
                                        response_text.append(gemini_message.text)
                                    if gemini_message.server_content and gemini_message.server_content.turn_complete:
                                        break

                                # Parse Gemini's JSON response
                                try:
                                    gemini_output = json.loads("".join(response_text))
                                    response = gemini_output.get("response", "Sorry, something went wrong.")
                                    state = validate_state(gemini_output.get("state", state))
                                    logger.info(f"Gemini response: {''.join(response_text)}")
                                except json.JSONDecodeError:
                                    response = "Sorry, something went wrong. Please try again."
                                    state = state
                            except Exception as e:
                                await websocket.send(json.dumps({
                                    "error": f"Gemini API error: {str(e)}",
                                    "session_id": session_id
                                }))
                                return

                        # Prepare n8n payload
                        n8n_payload = {
                            "body": {
                                "message": user_input,
                                "session_id": data.get("session_id", session_id),
                                "response": response,
                                "state": state,
                                "timestamp": datetime.now(uae_tz).strftime("%Y-%m-%d %H:%M:%S")
                            },
                            "headers": {"authorization": data.get("authorization", "")}
                        }

                        # Send to n8n
                        n8n_response = await call_n8n_webhook(n8n_payload)
                        logger.info(f"n8n response: {n8n_response}")

                        # Send response to frontend
                        frontend_response = {
                            "response": response,
                            "session_id": session_id,
                            "state": state
                        }
                        await websocket.send(json.dumps(frontend_response))

                    except json.JSONDecodeError:
                        await websocket.send(json.dumps({
                            "error": "Invalid JSON format: Please send a valid JSON object",
                            "session_id": session_id
                        }))
                    except requests.RequestException as e:
                        await websocket.send(json.dumps({
                            "error": f"Webhook error: {str(e)}",
                            "session_id": session_id
                        }))
                    except Exception as e:
                        await websocket.send(json.dumps({
                            "error": f"Unexpected error: {str(e)}",
                            "session_id": session_id
                        }))

    except asyncio.TimeoutError:
        await websocket.close(code=1000, reason="Inactivity timeout")
    except websockets.exceptions.ConnectionClosed:
        pass  # Handle client disconnect gracefully
    except Exception as e:
        await websocket.send(json.dumps({"error": str(e), "session_id": session_id}))

async def main():
    async with websockets.serve(handle_websocket, "0.0.0.0", 8765):
        print("WebSocket server started on ws://0.0.0.0:8765")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())