import asyncio
import websockets
import json
import requests
from dotenv import load_dotenv
import os
from datetime import datetime
from google import genai
from google.genai import types
from google.genai.types import Tool
import uuid
import logging
import base64
import io
import soundfile as sf
import librosa
import pytz

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

process_ride_details = {
    "name": "process_ride_details",
    "description": "Processes ride booking details to fetch service availaibility / fare / time slots. To be called immidiately after collecting start location , end location , start time , date .",
    "parameters": {
        "type": "object",
        "properties": {
            "startLocation": {"type": "string", "description": "The starting location of the ride."},
            "endLocation": {"type": "string", "description": "The destination location of the ride."},
            "startDate": {"type": "string", "description": "The date of the ride in DD-MM-YYYY format."},
            "startTime": {"type": "string", "description": "The time of the ride in H:MM AM/PM format."},
            "rideConfirmation": {"type": "boolean", "description": "The confirmation of the ride."},
        },
        "required": ["startLocation", "endLocation", "startDate", "startTime"]
    }
}

book_ride = {
    "name": "book_ride",
    "description": "Books a ride with the provided details. Only call this function when the user confirms to book the ride after being presented with the fare.",
    "parameters": {
        "type": "object",
        "properties": {
            "rideConfirmation": {"type": "boolean", "description": "The confirmation of the ride."},
            "session_id": {"type": "string", "description": "The session id , fetched after the fare is calculated."},
        },
        "required": ["rideConfirmation","session_id"]
    }
}



async def call_n8n_webhook(data):
    """Send structured output to n8n webhook"""
    headers = {"Content-Type": "application/json"}
    response = requests.post(N8N_WEBHOOK_URL, json=data, headers=headers)
    response.raise_for_status()
    return response.json()


async def handle_websocket(websocket):
    dubai_tz = pytz.timezone("Asia/Dubai")
    now_in_dubai = datetime.now(dubai_tz)
    current_dubai_time = now_in_dubai.strftime("%I:%M %p")
    current_dubai_date = now_in_dubai.strftime("%d-%m-%Y")
    logger.info(f"Current Dubai time: {current_dubai_time}, date: {current_dubai_date}")

    SYSTEM_PROMPT = f"""You are Tala, a friendly and knowledgeable AI assistant designed for users in the UAE. You handle general conversations, answer questions, provide relevant recommendations and book rides if requested, tailoring responses to the UAE context where appropriate. 
Current date is {current_dubai_date}. Current time is {current_dubai_time}. Use these for all date and time references in responses.
Use conversation history to maintain context and avoid repeating questions unnecessarily. Respond only in English.

Ride Booking Flow:
- If the user expresses intent to book a ride, collect ride details one at a time : 
    - start location
    - end location
    - start time
    - start date (default to {current_dubai_date}, unless specified otherwise by the user on their own)

- Once all required details (start location, end location, date, and start time) are collected, immediately call the 'process_ride_details' function.
- Use the function response to provide a friendly, clear summary of the details provided by the function. 
- If time slots are returned, ask the user to select one, then update the start time with the chosen slot.
- Only when the user confirms to book ride AFTER being presented with the fare , proceed to call the 'process_ride_details' function with the ride confirmation set to true.

Ride booking guielines:
- For airport-related locations, clarify which airport and terminal (e.g., Dubai International Airport, Terminal 1, 2, or 3).
- Only use the backend response to provide fares and time slots.
- If the user provides conflicting information, prioritize the most recent details provided.
- Maintain a conversational tone, responding to all user queries in a friendly, engaging and relevant manner.
"""
    session_id = f"{int(asyncio.get_event_loop().time())}-{uuid.uuid4().hex[:8]}"
    state = {
        "startLocation": None,
        "endLocation": None,
        "startDate": None,
        "startTime": None,
        "rideConfirmation": False,
    }
    tools = types.Tool(function_declarations=[process_ride_details, book_ride])

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
                            await websocket.send(json.dumps({"error": "No input received","session_id": session_id}))
                            continue
                        if not data.get("authorization"):
                            await websocket.send(json.dumps({"error": "Missing authorization","session_id": session_id}))
                            continue
                        if audio_input:
                            print("Audio input recieved")
                            try:
                                audio_bytes = base64.b64decode(audio_input)
                                buffer = io.BytesIO(audio_bytes)
                                y, sr = librosa.load(buffer, sr=16000, mono=True)
                                pcm_buffer = io.BytesIO()
                                sf.write(pcm_buffer, y, sr, format='RAW', subtype='PCM_16')
                                pcm_buffer.seek(0)
                                await session.send_realtime_input(audio=types.Blob(data=pcm_buffer.read(), mime_type="audio/pcm;rate=16000"))
                                print("Audio Sent")
                            except Exception as e:
                                await websocket.send(json.dumps({"error": f"Audio processing error: {str(e)}","session_id": session_id}))
                                continue
                        else:
                            print("Text input recieved")
                            await session.send_client_content(turns={"role": "user", "parts": [{"text": user_input}]}, turn_complete=True)
                        full_response_text = ""
                        gemini_transcription = user_input or ""

                        async for gemini_message in session.receive():
                            if gemini_message.text:
                                await websocket.send(json.dumps({
                                    "type": "chunk", 
                                    "response_chunk": gemini_message.text,
                                    "session_id": session_id,
                                }))

                            if gemini_message.server_content and gemini_message.server_content.input_transcription:
                                gemini_transcription += gemini_message.server_content.input_transcription.text
                                await websocket.send(json.dumps({
                                    "type": "chunk", 
                                    "transcription_chunk": gemini_message.server_content.input_transcription.text,
                                    "session_id": session_id,
                                }))
                            if gemini_message.tool_call:
                                function_responses = []
                                for fc in gemini_message.tool_call.function_calls:
                                    try:
                                        if fc.name == "process_ride_details":
                                            params = fc.args
                                            state.update(params)
                                            try:
                                                if state.get("startDate"): datetime.strptime(state["startDate"], "%d-%m-%Y")
                                                if state.get("startTime"): datetime.strptime(state["startTime"], "%I:%M %p")
                                            except ValueError as e:
                                                logger.error(f"Invalid state format: {str(e)}")
                                                function_responses.append(types.FunctionResponse(id=fc.id,name=fc.name,response={"error": f"Invalid date/time format: {str(e)}"}))
                                                continue
                                            
                                            n8n_payload = {"message": gemini_transcription,"session_id": session_id,"state": state,"headers": {"authorization": data.get("authorization", "")}}
                                            n8n_response = await call_n8n_webhook(n8n_payload)
                                            if "state" in n8n_response:
                                                state.update(n8n_response["state"])

                                            function_responses.append(types.FunctionResponse(
                                                id=fc.id,
                                                name=fc.name,
                                                response=n8n_response
                                            ))
                                        elif fc.name == "book_ride":
                                            params = fc.args
                                            state.update(params)
                                            n8n_payload = {"session_id": session_id,"state": state,"headers": {"authorization": data.get("authorization", "")}}
                                            n8n_response = await call_n8n_webhook(n8n_payload)

                                            function_responses.append(types.FunctionResponse(
                                                id=fc.id,
                                                name=fc.name,
                                                response=n8n_response
                                            ))
                                    except Exception as e:
                                        logger.error(f"Error processing function call '{fc.name}': {str(e)}")
                                        function_responses.append(types.FunctionResponse(id=fc.id,name=fc.name,response={"error": str(e)}))

                                await session.send_tool_response(function_responses=function_responses)

                        await websocket.send(json.dumps({
                            "type": "final", 
                            "session_id": session_id,
                        }))

                    except json.JSONDecodeError:
                        await websocket.send(json.dumps({"error": "Invalid JSON format: Please send a valid JSON object","session_id": session_id}))
                    except requests.RequestException as e:
                        await websocket.send(json.dumps({"error": f"Webhook error: {str(e)}","session_id": session_id}))
                    except Exception as e:
                        await websocket.send(json.dumps({"error": f"Unexpected error: {str(e)}","session_id": session_id}))

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