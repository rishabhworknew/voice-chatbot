import asyncio
import websockets
import json
from dotenv import load_dotenv
import os
from datetime import datetime
from google import genai
from google.genai import types
import uuid
import logging
import base64
import pytz
from config import (
    get_system_prompt,
    get_fare_details,
    book_ride,
    call_n8n_webhook,
    reverse_geocode
)

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# --- API and Environment Configuration ---
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is missing!")

client = genai.Client(api_key=API_KEY)
model_id = "gemini-live-2.5-flash-preview"


async def handle_websocket(websocket):
    dubai_tz = pytz.timezone("Asia/Dubai")
    now_in_dubai = datetime.now(dubai_tz)
    current_dubai_time = now_in_dubai.strftime("%I:%M %p")
    current_dubai_date = now_in_dubai.strftime("%d-%m-%Y")
    logger.info(f"Current Dubai time: {current_dubai_time}, date: {current_dubai_date}")

    session_id = f"{int(asyncio.get_event_loop().time())}-{uuid.uuid4().hex[:8]}"
    
    # Initialize session state
    state = {
        "startLocation": None,
        "endLocation": None,
        "startDate": None,
        "startTime": None,
        "rideConfirmation": False,
        "authorization_token": None,
    }
    
    tools = types.Tool(function_declarations=[get_fare_details, book_ride])
    logger.info(f"New client connection established with session ID: {session_id}")

    try:
        # --- Authentication Step ---
        auth_message = await websocket.recv()
        auth_data = json.loads(auth_message)

        if auth_data.get("type") == "auth" and auth_data.get("token"):
            state["authorization_token"] = auth_data["token"]
            state["user_name"] = auth_data.get("name", "Unknown")
            state["latitude"] = auth_data.get("latitude", "Unknown")
            state["longitude"] = auth_data.get("longitude", "Unknown")
            
            if state["latitude"] != "Unknown" and state["longitude"] != "Unknown":
                # state["address"] = await reverse_geocode(state["latitude"], state["longitude"])
                state["address"] = "Dubai Marina" # Using a placeholder as in the original code
            else:
                state["address"] = "Unknown location"
                
            logger.info(f"Authentication successful for user: {state['user_name']}")
            await websocket.send(json.dumps({"type": "auth_status", "status": "success", "session_id": session_id}))
        else:
            logger.warning("First message was not a valid authentication message. Closing connection.")
            await websocket.send(json.dumps({"error": "Authentication required as first message.", "session_id": session_id}))
            await websocket.close()
            return

        # --- Gemini Configuration ---
        system_prompt_text = get_system_prompt(state, current_dubai_date, current_dubai_time)
        
        config = types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            input_audio_transcription={},
            output_audio_transcription={},
            system_instruction=types.Content(parts=[types.Part(text=system_prompt_text)]),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Kore"
                    )
                )
            ),
            tools=[tools]
        )

        # --- Main Communication Loop ---
        async with client.aio.live.connect(model=model_id, config=config) as session:
            
            async def gemini_to_client():
                try:
                    booking_confirmed = False
                    while True:
                        async for gemini_message in session.receive():
                            # Handle audio data
                            if gemini_message.data is not None:
                                pcm_data = base64.b64decode(gemini_message.data)
                                audio_chunk_b64 = base64.b64encode(pcm_data).decode('utf-8')
                                await websocket.send(json.dumps({
                                    "type": "audio_chunk", "audio_chunk": audio_chunk_b64, "session_id": session_id
                                }))

                            # Handle transcriptions
                            if gemini_message.server_content:
                                if gemini_message.server_content.input_transcription:
                                    await websocket.send(json.dumps({
                                        "type": "chunk", "transcription_chunk": gemini_message.server_content.input_transcription.text, "session_id": session_id
                                    }))
                                    logger.info(f"Transcription: {gemini_message.server_content.input_transcription.text}")
                                if gemini_message.server_content.output_transcription:
                                    await websocket.send(json.dumps({
                                        "type": "chunk", "response_chunk": gemini_message.server_content.output_transcription.text, "session_id": session_id
                                    }))
                                    logger.info(f"Response: {gemini_message.server_content.output_transcription.text}")

                            # Handle tool calls
                            if gemini_message.tool_call:
                                logger.info(f"Tool call received: {gemini_message.tool_call}")
                                function_responses = []
                                for fc in gemini_message.tool_call.function_calls:
                                    try:
                                        if fc.name == "get_fare_details":
                                            state.update(fc.args)
                                            # Validate date/time format before making the webhook call
                                            try:
                                                if state.get("startDate"): datetime.strptime(state["startDate"], "%d-%m-%Y")
                                                if state.get("startTime"): datetime.strptime(state["startTime"], "%I:%M %p")
                                            except ValueError as e:
                                                logger.error(f"Invalid date/time format in state: {e}")
                                                function_responses.append(types.FunctionResponse(id=fc.id, name=fc.name, response={"error": f"Invalid date/time format: {e}"}))
                                                continue
                                            
                                            n8n_payload = {"session_id": session_id, "state": state, "headers": {"authorization": state.get("authorization_token", "")}}
                                            n8n_response = await call_n8n_webhook(n8n_payload)
                                            state["fare"] = n8n_response.get("fare")
                                            if "state" in n8n_response:
                                                state.update(n8n_response["state"])
                                            function_responses.append(types.FunctionResponse(id=fc.id, name=fc.name, response=n8n_response))

                                        elif fc.name == "book_ride":
                                            state.update(fc.args)
                                            n8n_payload = {"session_id": session_id, "state": state, "headers": {"authorization": state.get("authorization_token", "")}}
                                            n8n_response = await call_n8n_webhook(n8n_payload)
                                            if n8n_response.get("status") == "BOOKING_CONFIRMED":
                                                booking_confirmed = True
                                            function_responses.append(types.FunctionResponse(id=fc.id, name=fc.name, response=n8n_response))
                                    
                                    except Exception as e:
                                        logger.error(f"Error processing function call '{fc.name}': {e}")
                                        function_responses.append(types.FunctionResponse(id=fc.id, name=fc.name, response={"error": str(e)}))
                                
                                await session.send_tool_response(function_responses=function_responses)

                            if gemini_message.server_content and gemini_message.server_content.turn_complete:
                                await asyncio.sleep(0.5)
                                await websocket.send(json.dumps({"type": "final", "session_id": session_id}))
                                logger.info("Gemini turn complete.")
                                if booking_confirmed:
                                    logger.info("Booking confirmed, sending final confirmation to client.")
                                    await websocket.send(json.dumps({"type": "confirm"}))
                                    break # End the loop after confirmation

                except websockets.exceptions.ConnectionClosed:
                    logger.info("Connection closed while streaming from Gemini.")
                except Exception as e:
                    logger.error(f"Error in gemini_to_client task: {e}")
                    await websocket.send(json.dumps({"error": str(e), "session_id": session_id}))

            async def client_to_gemini():
                try:
                    async for message in websocket:
                        data = json.loads(message)
                        user_input = data.get("text")
                        audio_input = data.get("audio")
                        
                        if user_input:
                            await session.send_client_content(turns={"role": "user", "parts": [{"text": user_input}]}, turn_complete=True)
                            logger.info(f"Client sent text to Gemini: {user_input}")
                        elif audio_input:
                            audio_bytes = base64.b64decode(audio_input)
                            await session.send_realtime_input(
                                audio=types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000")
                            )
                except websockets.exceptions.ConnectionClosed:
                    logger.info("Client connection closed.")
                except Exception as e:
                    logger.error(f"Error in client_to_gemini task: {e}")
                    await websocket.send(json.dumps({"error": str(e), "session_id": session_id}))

            # Run both tasks concurrently and wait for them to complete
            await asyncio.gather(gemini_to_client(), client_to_gemini())
            logger.info(f"Session {session_id} ended.")

    except Exception as e:
        logger.error(f"Overall websocket error: {e}")
        try:
            await websocket.send(json.dumps({"error": str(e), "session_id": session_id}))
        except websockets.exceptions.ConnectionClosed:
            pass # Client already disconnected

async def main():
    """Starts the WebSocket server."""
    port = int(os.getenv("PORT", 8765))
    async with websockets.serve(handle_websocket, "0.0.0.0", port):
        logger.info(f"WebSocket server started on ws://0.0.0.0:{port}")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
