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
    "description": "Processes ride booking details to fetch service availability / fare / time slots. To be called immediately after collecting start location, end location, start time, date.",
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
            "session_id": {"type": "string", "description": "The session id, fetched after the fare is calculated."},
        },
        "required": ["rideConfirmation", "session_id"]
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

    SYSTEM_PROMPT = """
You are a versatile and friendly AI assistant."""
    session_id = f"{int(asyncio.get_event_loop().time())}-{uuid.uuid4().hex[:8]}"
    state = {
        "startLocation": None,
        "endLocation": None,
        "startDate": None,
        "startTime": None,
        "rideConfirmation": False,
    }
    tools = types.Tool(function_declarations=[process_ride_details, book_ride])

    logger.info("New client connection established.")

    try:
        config = types.LiveConnectConfig(
            response_modalities=[types.Modality.TEXT],
            input_audio_transcription={},
            system_instruction=types.Content(parts=[types.Part(text=SYSTEM_PROMPT)]),
            tools=[tools]
        )

        async with client.aio.live.connect(model=model_id, config=config) as session:
            # TASK 1: Receive from Gemini and send to Client
            async def gemini_to_client():
                try:
                    while True:
                        async for gemini_message in session.receive():
                            if gemini_message.text:
                                await websocket.send(json.dumps({
                                    "type": "chunk", "response_chunk": gemini_message.text, "session_id": session_id
                                }))
                            print("Gemini sent text:", gemini_message.text)

                            if gemini_message.server_content and gemini_message.server_content.input_transcription:
                                await websocket.send(json.dumps({
                                    "type": "chunk", "transcription_chunk": gemini_message.server_content.input_transcription.text, "session_id": session_id
                                }))
                                print("Gemini sent transcription:", gemini_message.server_content.input_transcription.text)

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
                                                function_responses.append(types.FunctionResponse(id=fc.id, name=fc.name, response={"error": f"Invalid date/time format: {str(e)}"}))
                                                continue

                                            n8n_payload = {"message": gemini_message.text or "", "session_id": session_id, "state": state, "headers": {"authorization": data.get("authorization", "")}}
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
                                            n8n_payload = {"session_id": session_id, "state": state, "headers": {"authorization": data.get("authorization", "")}}
                                            n8n_response = await call_n8n_webhook(n8n_payload)

                                            function_responses.append(types.FunctionResponse(
                                                id=fc.id,
                                                name=fc.name,
                                                response=n8n_response
                                            ))
                                    except Exception as e:
                                        logger.error(f"Error processing function call '{fc.name}': {str(e)}")
                                        function_responses.append(types.FunctionResponse(id=fc.id, name=fc.name, response={"error": str(e)}))

                                await session.send_tool_response(function_responses=function_responses)

                            if gemini_message.server_content and gemini_message.server_content.turn_complete:
                                await websocket.send(json.dumps({"type": "final", "session_id": session_id}))
                                print("Gemini turn complete.")
                                break

                except websockets.exceptions.ConnectionClosed:
                    logger.info("Connection closed while streaming from Gemini.")
                except Exception as e:
                    logger.error(f"Error in gemini_to_client task: {e}")
                    await websocket.send(json.dumps({"error": str(e), "session_id": session_id}))

            # TASK 2: Receive from Client and send to Gemini
            async def client_to_gemini():
                try:
                    async for message in websocket:
                        data = json.loads(message)
                        user_input = data.get("text")
                        audio_input = data.get("audio")
                        if user_input:
                            print("Client sent text:", user_input)
                            await session.send_client_content(turns={"role": "user", "parts": [{"text": user_input}]}, turn_complete=True)
                            print("Client sent text to Gemini.")

                        elif audio_input:
                            print("Client sent audio:", audio_input)
                            audio_bytes = base64.b64decode(audio_input)
                            await session.send_realtime_input(
                                audio=types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000")
                            )
                            print("Client sent audio to Gemini.")

                        elif data.get("action") == "audio_input_ended": # New condition
                            print("Client signaled end of audio input for the turn.")
                            await session.send_realtime_input(audio_stream_end=True)
                            print("Client signaled end of audio input for the turn to Gemini.")

                except websockets.exceptions.ConnectionClosed:
                    logger.info("Client connection closed.")
                except Exception as e:
                    logger.error(f"Error in client_to_gemini task: {e}")
                    await websocket.send(json.dumps({"error": str(e), "session_id": session_id}))

            # Run both tasks concurrently
            await asyncio.gather(gemini_to_client(), client_to_gemini())
    
            print("Session ended.")

    except Exception as e:
        logger.error(f"Overall websocket error: {e}")
        try:
            await websocket.send(json.dumps({"error": str(e), "session_id": session_id}))
        except websockets.exceptions.ConnectionClosed:
            pass

async def main():
    async with websockets.serve(handle_websocket, "0.0.0.0", 8765):
        print("WebSocket server started on ws://0.0.0.0:8765")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())