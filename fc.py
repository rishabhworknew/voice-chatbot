import asyncio
import websockets
import json
import requests
from dotenv import load_dotenv
import os
from datetime import datetime
from google import genai
from google.genai import types
import uuid
import logging
import base64
import io
import soundfile as sf
import librosa
import pytz

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
    dubai_tz = pytz.timezone("Asia/Dubai")
    dubai_time = datetime.now(dubai_tz)
    return dubai_time.strftime("%I:%M %p")

def get_dubai_date():
    dubai_tz = pytz.timezone("Asia/Dubai")
    dubai_time = datetime.now(dubai_tz)
    return dubai_time.strftime("%d-%m-%Y")

current_dubai_time = get_dubai_time()
current_dubai_date = get_dubai_date()
logger.info(f"Current Dubai time: {current_dubai_time}, date: {current_dubai_date}")

# System prompt
SYSTEM_PROMPT = f"""
You are Tala, a sophisticated and friendly AI assistant for users in the UAE. Your personality is that of a knowledgeable concierge.
You must handle all general conversation, answer questions, and provide recommendations if the user requests it. Respond in a friendly, interactive manner.
Use conversation history maintain context and avoid asking for the same information multiple times.


Accept any language from the user but respond only in English. 
Current dubai date : {current_dubai_date}. Current dubai time : {current_dubai_time}. Use this for all date and time related references.
Assume today’s date unless the user specifies otherwise.
Task:
- If the user wants to book a ride, ask for one detail at a time: start location, end location, date (default today), start time.
- When you have this information, you will call the `process_ride_details` function immidiately. It provides fare and time slots. Use those backend details to provide a friendly response to the user.
- For airport locations, clarify which airport and terminal the user is referring to.
- If the backend provides availaible time slots, ask the user to select a time slot. startTime is the users requested time,selectedSlot is the time slot availaible from the backend. 
- If the user provides conflicting information, always use the latest information.

"""

# Function declaration
process_ride_details = {
    "name": "process_ride_details",
    "description": "Processes ride booking details to fetch fare and available time slots.",
    "parameters": {
        "type": "object",
        "properties": {
            "startLocation": {"type": "string", "description": "The starting location of the ride."},
            "endLocation": {"type": "string", "description": "The destination location of the ride."},
            "startDate": {"type": "string", "description": "The date of the ride in DD-MM-YYYY format."},
            "startTime": {"type": "string", "description": "The time of the ride in H:MM AM/PM format."},
            "selectedSlot": {"type": "string", "description": "The selected time slot for the ride."},
            "rideConfirmation": {"type": "boolean", "description": "The confirmation of the ride."},
        },
        "required": ["startLocation", "endLocation", "startDate", "startTime"]
    }
}

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
    tools = types.Tool(function_declarations=[process_ride_details])

    try:
        async with asyncio.timeout(600):
            config = types.LiveConnectConfig(
                response_modalities=[types.Modality.TEXT],
                input_audio_transcription={},
                system_instruction=types.Content(parts=[types.Part(text=SYSTEM_PROMPT)]),
                tools=[tools]
            )

            async with client.aio.live.connect(model=model_id, config=config) as session:
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        user_input = data.get("text")
                        audio_input = data.get("audio")

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
                        # Handle audio input

                        if audio_input:
                            print("Audio input received")
                            try:
                                audio_bytes = base64.b64decode(audio_input)
                                buffer = io.BytesIO(audio_bytes)
                                y, sr = librosa.load(buffer, sr=16000, mono=True)
                                pcm_buffer = io.BytesIO()
                                sf.write(pcm_buffer, y, sr, format='RAW', subtype='PCM_16')
                                pcm_buffer.seek(0)

                                await session.send_realtime_input(
                                    audio=types.Blob(data=pcm_buffer.read(), mime_type="audio/pcm;rate=16000")
                                )
                            except Exception as e:
                                await websocket.send(json.dumps({
                                    "error": f"Audio processing error: {str(e)}",
                                    "session_id": session_id
                                }))
                                continue
                        else:
                            print("Text input received" + user_input)
                            await session.send_client_content(
                                turns={
                                    "role": "user",
                                    "parts": [{"text": user_input}]
                                },
                                turn_complete=True
                            )
                        full_response_text = ""
                        transcription = user_input or ""
                        async for gemini_message in session.receive():
                            if gemini_message.text:
                                full_response_text += gemini_message.text
                            
                            if gemini_message.server_content and gemini_message.server_content.input_transcription:
                                transcription += gemini_message.server_content.input_transcription.text
                                print("Transcription: " + transcription)

                            if gemini_message.tool_call:
                                function_responses = []
                                print("########################")
                                print(gemini_message)
                                print("########################")

                                for fc in gemini_message.tool_call.function_calls:
                                    if fc.name == "process_ride_details":
                                        try:
                                            params = fc.args
                                            state.update(params)
                                            print(params)
                                            print(state)
                                            # Basic validation
                                            try:
                                                if state.get("startDate"):
                                                    datetime.strptime(state["startDate"], "%d-%m-%Y")
                                                if state.get("startTime"):
                                                    datetime.strptime(state["startTime"], "%I:%M %p")
                                            except ValueError as e:
                                                logger.error(f"Invalid state format: {str(e)}")
                                                function_responses.append(types.FunctionResponse(
                                                    id=fc.id,
                                                    name=fc.name,
                                                    response={"error": f"Invalid state format: {str(e)}"}
                                                ))
                                                continue

                                            n8n_payload = {
                                                "message": user_input,
                                                "session_id": session_id,
                                                "state": state,
                                                "timestamp": datetime.now(uae_tz).isoformat(),
                                                "headers": {"authorization": data.get("authorization", "")}
                                            }

                                            n8n_response = await call_n8n_webhook(n8n_payload)
                                            if "state" in n8n_response:
                                                state.update(n8n_response["state"])
                                            print("#####################")
                                            print(n8n_response)
                                            print("#####################")

                                            function_responses.append(types.FunctionResponse(
                                                id=fc.id,
                                                name=fc.name,
                                                response=n8n_response
                                            ))
                                            print(function_responses)
                                        except Exception as e:
                                            logger.error(f"Error processing function call: {str(e)}")
                                            function_responses.append(types.FunctionResponse(
                                                id=fc.id,
                                                name=fc.name,
                                                response={"error": str(e)}
                                            ))
                                
                                # Send function responses back to Gemini
                                await session.send_tool_response(function_responses=function_responses)

                        if full_response_text:
                            await websocket.send(json.dumps({
                                "response": full_response_text.strip(),
                                "session_id": session_id,
                                "state": state,
                                "transcription": transcription
                            }))

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
        pass
    except Exception as e:
        await websocket.send(json.dumps({"error": str(e), "session_id": session_id}))

async def main():
    async with websockets.serve(handle_websocket, "0.0.0.0", 8765):
        print("WebSocket server started on ws://0.0.0.0:8765")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())