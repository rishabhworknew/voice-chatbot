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

def get_dubai_time():
    utc_time = datetime.now(UTC)
    dubai_time = utc_time + timedelta(hours=4) 
    formatted_time = dubai_time.strftime("%I:%M %p")
    return formatted_time

def get_dubai_date():
    utc_time = datetime.now(UTC)
    dubai_time = utc_time + timedelta(hours=4) 
    formatted_time = dubai_time.strftime("%d-%m-%Y")
    return formatted_time
current_dubai_time = get_dubai_time()
current_dubai_date = get_dubai_date()
# System prompt for ride-booking assistant
#logger.info(f"Current Dubai time : {current_dubai_time}")
#logger.info(f"Current Dubai date : {current_dubai_date}")
SYSTEM_PROMPT = f"""
You are a ride-booking assistant in the UAE. Current dubai date : {current_dubai_date}. Current dubai time : {current_dubai_time}. 
Assume today’s date unless the user specifies otherwise.

Conversational Task:
- Guide the user to book a ride by asking for one detail at a time: start location, end location, date (default today), time.
- Respond concisely, friendly, and interactively. If the user asks unrelated questions, gently redirect to booking after acknowledging the question.
- If the user wants to search for nearby locations, respond with the list of locations.

State Extraction:
Extract and update these fields based on the user’s message and history:
- "startLocation": string (e.g., "Dubai Airport")
- "endLocation": string (e.g., "Dubai Mall")
- "startDate": string (DD-MM-YYYY, default today)
- "startTime": string (H:MM AM/PM, e.g., "2:00 PM")
- "selectedSlot": string (e.g., "02:00 PM") if user selects a slot
- "rideConfirmation": boolean (true if user says "yes" after fare)
- "rideRejection": boolean (true if user says "no" after fare)
Update fields with new information, keeping prior values unless contradicted.

Output:
Return a JSON object:
{{
  "response": "Conversational response",
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
                                    gemini_output = json.loads(json_to_parse.strip()) # Use the cleaned string
                                    response = gemini_output.get("response", "Sorry, something went wrong with the expected output format.")
                                    state_update = gemini_output.get("state")
                                    if state_update is not None:
                                        state = validate_state(state_update)
                                    else:
                                        # Keep current state if "state" key is missing in Gemini's output
                                        pass # Or handle as an error if state is always expected
                                    logger.info(f"Successfully parsed Gemini JSON: {gemini_output}")
                                except json.JSONDecodeError as e:
                                    logger.error(f"JSONDecodeError: {e}. String attempted for parsing: '{json_to_parse.strip()}'")
                                    response = "Sorry, something went wrong processing the response. Please try again."
                                    # state remains unchanged from its previous value
                                except ValueError as e: # Catch validation errors
                                    logger.error(f"State validation error: {e}. Offending state: {state_update}")
                                    response = f"Sorry, there was an issue with the data format: {e}"
                                    # state remains unchanged or you might want to revert
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

                                # Process Gemini response# Process Gemini response
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
                                    gemini_output = json.loads(json_to_parse.strip()) # Use the cleaned string
                                    response = gemini_output.get("response", "Sorry, something went wrong with the expected output format.")
                                    state_update = gemini_output.get("state")
                                    if state_update is not None:
                                        state = validate_state(state_update)
                                    else:
                                        # Keep current state if "state" key is missing in Gemini's output
                                        pass # Or handle as an error if state is always expected
                                    logger.info(f"Successfully parsed Gemini JSON: {gemini_output}")
                                except json.JSONDecodeError as e:
                                    logger.error(f"JSONDecodeError: {e}. String attempted for parsing: '{json_to_parse.strip()}'")
                                    response = "Sorry, something went wrong processing the response. Please try again."
                                    # state remains unchanged from its previous value
                                except ValueError as e: # Catch validation errors
                                    logger.error(f"State validation error: {e}. Offending state: {state_update}")
                                    response = f"Sorry, there was an issue with the data format: {e}"
                                    # state remains unchanged or you might want to revert
                            except Exception as e:
                                await websocket.send(json.dumps({
                                    "error": f"Gemini API error: {str(e)}",
                                    "session_id": session_id
                                }))
                                return

                        # Prepare n8n payload
                        n8n_payload = {
                                "message": user_input,
                                "session_id": data.get("session_id", session_id),
                                "response": response,
                                "state": state,
                                "timestamp": datetime.now(uae_tz).strftime("%Y-%m-%d %H:%M:%S"),
                                "headers": {"authorization": data.get("authorization", "")}
                        }

                        # Send to n8n
                        n8n_processed_data = await call_n8n_webhook(n8n_payload) # This is the JSON dict from n8n
                        logger.info(f"Received from n8n: {n8n_processed_data}")

                        # Send response to frontend
                        final_user_response = n8n_processed_data.get("response", "Sorry, there was an issue processing your request with our backend services.")
                        final_state = n8n_processed_data.get("state", state) # Fallback to previous state if n8n doesn't send one

                        frontend_response = {
                            "response": final_user_response, # Use response from n8n
                            "session_id": session_id,
                            "state": final_state             # Use state from n8n (or updated by it)
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