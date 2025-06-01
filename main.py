import asyncio
import websockets
import json
import requests
from dotenv import load_dotenv
import os
from datetime import datetime, UTC, timedelta
import pytz
from tenacity import retry, stop_after_attempt, wait_fixed
from google import genai
from google.genai import types
import uuid
import logging
import re
import io
import soundfile as sf
import librosa
import base64

# Configure logging - FINAL
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

def get_dubai_time():
    dubai_tz = pytz.timezone("Asia/Dubai")
    dubai_time = datetime.now(dubai_tz)
    return dubai_time.strftime("%I:%M %p")

def get_dubai_date():
    dubai_tz = pytz.timezone("Asia/Dubai")
    dubai_time = datetime.now(dubai_tz)
    return dubai_time.strftime("%d-%m-%Y")

current_dubai_time = get_dubai_time()
current_dubai_date = get_dubai_date()
logger.info(f"Current Dubai time : {current_dubai_time} , date : {current_dubai_date}")
SYSTEM_PROMPT = f"""
You are Tala , a general purpose assistant in the UAE that can also book a ride. Accept any language from the user but respond only in English. 
Current dubai date : {current_dubai_date}. Current dubai time : {current_dubai_time}. 
Assume today’s date unless the user specifies otherwise.

Conversational Task:
- If the user wants to book a ride, ask for one detail at a time: start location, end location, date (default today), start time.
- If the user asks unrelated questions, gently redirect to booking after acknowledging the question.
- If the user queries about location suggestions/reccomendations , answer their question.

State Extraction:
Extract and update these fields based on the user’s message and history:
- "startLocation": string (e.g., "Dubai Airport Terminal 1")
- "endLocation": string (e.g., "Dubai Mall")
- "startDate": string (DD-MM-YYYY, default today)
- "startTime": string (H:MM AM/PM, e.g., "2:00 PM")
- "selectedSlot": string (e.g., "02:00 PM") if user selects a slot
- "rideConfirmation": boolean (true if user says "yes" after fare)
- "rideRejection": boolean (true if user says "no" after fare)
Update fields with new information, keeping prior values unless contradicted.

Output:
Always return a JSON object in this format:
{{
    "transcription": "Transcription of the user's message",
    "response": "Your concise, friendly, and interactive response",
    "state": {{ "startLocation": null, "endLocation": null, "startDate": null, "startTime": null, "selectedSlot": null, "rideConfirmation": false, "rideRejection": false }}
}}
"""
# logger.info(f"System prompt: {SYSTEM_PROMPT}")

def validate_state(state):
    """Validate state fields"""
    try:
        if state["startDate"]:
            datetime.strptime(state["startDate"], "%d-%m-%Y")
        if state["startTime"]:
            datetime.strptime(state["startTime"], "%I:%M %p")
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
        async with asyncio.timeout(600):  # 10-minute timeout for inactivity
            # Configure Live API session with system instruction (remove input_audio_transcription)
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
                        logger.info(f"User input: {user_input}")
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

                        # Initialize variables
                        transcription = ""
                        response = ""
                        state_update = None

                        if audio_input:
                            try:
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
                                response_text_parts = []
                                async for gemini_message in session.receive():
                                    if gemini_message.text:
                                        response_text_parts.append(gemini_message.text)
                                    if gemini_message.server_content and gemini_message.server_content.turn_complete:
                                        break

                                raw_gemini_response_str = "".join(response_text_parts)
                                logger.info(f"Raw Gemini response: {raw_gemini_response_str}")

                                # Extract JSON from markdown fences
                                json_to_parse = raw_gemini_response_str
                                match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_gemini_response_str)
                                if match:
                                    json_to_parse = match.group(1)

                                # Parse Gemini's JSON response
                                try:
                                    gemini_output = json.loads(json_to_parse.strip())
                                    transcription = gemini_output.get("transcription", "")
                                    response = gemini_output.get("response", "Sorry, something went wrong with the expected output format.")
                                    state_update = gemini_output.get("state")
                                    if state_update is not None:
                                        state = validate_state(state_update)
                                    logger.info(f"Successfully parsed Gemini JSON: {gemini_output}")
                                except json.JSONDecodeError as e:
                                    logger.error(f"JSONDecodeError: {e}. String attempted for parsing: '{json_to_parse.strip()}'")
                                    response = "Sorry, something went wrong processing the response. Please try again."
                                except ValueError as e:
                                    logger.error(f"State validation error: {e}. Offending state: {state_update}")
                                    response = f"Sorry, there was an issue with the data format: {e}"
                            except Exception as e:
                                await websocket.send(json.dumps({
                                    "error": f"Audio processing error: {str(e)}",
                                    "session_id": session_id
                                }))
                                continue
                        else:
                            # Handle text input
                            try:
                                await session.send_client_content(
                                    turns={
                                        "role": "user",
                                        "parts": [{"text": user_input}]
                                    },
                                    turn_complete=True
                                )

                                # Process Gemini response
                                response_text_parts = []
                                async for gemini_message in session.receive():
                                    if gemini_message.text:
                                        response_text_parts.append(gemini_message.text)
                                    if gemini_message.server_content and gemini_message.server_content.turn_complete:
                                        break

                                raw_gemini_response_str = "".join(response_text_parts)
                                logger.info(f"Raw Gemini response: {raw_gemini_response_str}")

                                # Extract JSON from markdown fences
                                json_to_parse = raw_gemini_response_str
                                match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_gemini_response_str)
                                if match:
                                    json_to_parse = match.group(1)

                                # Parse Gemini's JSON response
                                try:
                                    gemini_output = json.loads(json_to_parse.strip())
                                    transcription = gemini_output.get("transcription", user_input or "")  # Use user_input as fallback
                                    response = gemini_output.get("response", "Sorry, something went wrong with the expected output format.")
                                    state_update = gemini_output.get("state")
                                    if state_update is not None:
                                        state = validate_state(state_update)
                                    logger.info(f"Successfully parsed Gemini JSON: {gemini_output}")
                                except json.JSONDecodeError as e:
                                    logger.error(f"JSONDecodeError: {e}. String attempted for parsing: '{json_to_parse.strip()}'")
                                    response = "Sorry, something went wrong processing the response. Please try again."
                                except ValueError as e:
                                    logger.error(f"State validation error: {e}. Offending state: {state_update}")
                                    response = f"Sorry, there was an issue with the data format: {e}"
                            except Exception as e:
                                await websocket.send(json.dumps({
                                    "error": f"Gemini API error: {str(e)}",
                                    "session_id": session_id
                                }))
                                return

                        # Check if all required fields are fulfilled
                        required_fields = ["startLocation", "endLocation", "startDate", "startTime"]
                        all_fields_fulfilled = all(state[field] is not None and state[field].strip() != "" for field in required_fields)

                        # Prepare frontend response
                        frontend_response = {
                            "response": response,
                            "session_id": session_id,
                            "state": state,
                            "transcription": transcription
                        }
                        logger.info(f"Frontend response: {frontend_response}")

                        if all_fields_fulfilled:
                            # Prepare n8n payload
                            n8n_payload = {
                                "message": transcription or user_input,  # Prefer transcription, fallback to user_input
                                "session_id": data.get("session_id", session_id),
                                "response": response,
                                "state": state,
                                "transcription": transcription,
                                "timestamp": datetime.now(uae_tz).strftime("%Y-%m-%d %H:%M:%S"),
                                "headers": {"authorization": data.get("authorization", "")}
                            }
                            logger.info(f"N8N payload: {n8n_payload}")
                            # Send to n8n
                            try:
                                logger.info("All required fields fulfilled, sending to n8n webhook")
                                n8n_processed_data = await call_n8n_webhook(n8n_payload)
                                logger.info(f"Received from n8n: {n8n_processed_data}")

                                # Update frontend response with n8n data
                                frontend_response["response"] = n8n_processed_data.get("response", response)
                                logger.info(f"Updated response: {frontend_response['response']}")
                                frontend_response["state"] = n8n_processed_data.get("state", state)
                                logger.info(f"Updated state: {frontend_response['state']}")
                            except requests.RequestException as e:
                                logger.error(f"Webhook error: {str(e)}")
                                frontend_response["response"] = "Sorry, there was an issue contacting our backend services."
                        else:
                            logger.info("Not all required fields fulfilled, skipping n8n webhook call")

                        # Send response to frontend
                        await websocket.send(json.dumps(frontend_response))

                    except json.JSONDecodeError:
                        await websocket.send(json.dumps({
                            "error": "Invalid JSON format: Please send a valid JSON object",
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