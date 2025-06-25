import asyncio
import os
import json
import uuid
import logging
import base64
from datetime import datetime

# Use aiohttp for web server and websockets
from aiohttp import web
import aiohttp

# Other necessary libraries
from dotenv import load_dotenv
import pytz
import httpx
from google import genai
from google.genai import types

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# --- Environment Variable & API Configuration ---
API_KEY = os.getenv("GOOGLE_API_KEY")
PLACES_API_KEY = os.getenv("PLACES_API_KEY")
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")

if not API_KEY:
    raise RuntimeError("FATAL: GOOGLE_API_KEY environment variable is not set!")
if not N8N_WEBHOOK_URL:
    logger.warning("N8N_WEBHOOK_URL is not set. Tool calls requiring it will fail.")

# Initialize Google GenAI Client
try:
    client = genai.Client(api_key=API_KEY)
    model_id = "gemini-2.0-flash-live-001"
except Exception as e:
    logger.fatal(f"Failed to initialize Gemini client: {e}")
    raise

# --- Tool Definitions for Gemini ---
get_fare_details = {
    "name": "get_fare_details",
    "description": "Processes ride booking details to fetch service fare / time slots. All parameters to be sent in english.",
    "parameters": {
        "type": "object",
        "properties": {
            "startLocation": {"type": "string", "description": "The starting location of the ride."},
            "endLocation": {"type": "string", "description": "The destination location of the ride."},
            "startDate": {"type": "string", "description": "The date of the ride in DD-MM-YYYY format."},
            "startTime": {"type": "string", "description": "The time of the ride in H:MM AM/PM format, 12 hour format."},
            "rideConfirmation": {"type": "boolean", "description": "The confirmation of the ride."},
        },
        "required": ["startLocation", "endLocation", "startDate", "startTime"]
    }
}

book_ride = {
    "name": "book_ride",
    "description": "Books a ride with the provided details. Only call this function when the user confirms to book the ride after being presented with the fare returned by get_fare_details.",
    "parameters": {
        "type": "object",
        "properties": {
            "rideConfirmation": {"type": "boolean", "description": "The confirmation of the ride."},
            "fare": {"type": "string", "description": "The fare of the ride returned by get_fare_details."},
        },
        "required": ["rideConfirmation", "fare"]
    }
}

# --- Asynchronous Helper Functions ---
async def call_n8n_webhook(data):
    """Send structured output to the n8n webhook."""
    if not N8N_WEBHOOK_URL:
        logger.error("Cannot call n8n webhook because N8N_WEBHOOK_URL is not set.")
        return {"error": "Webhook URL not configured."}
    
    headers = {"Content-Type": "application/json"}
    async with httpx.AsyncClient() as http_client:
        try:
            response = await http_client.post(N8N_WEBHOOK_URL, json=data, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            logger.error(f"HTTP request to n8n webhook failed: {e}")
            return {"error": f"Failed to contact webhook service: {e}"}

# Note: reverse_geocode is defined but not used in the original logic. 
# It's kept here in case it's needed in the future.
async def reverse_geocode(lat, lon):
    """Performs reverse geocoding using Google Maps API."""
    if not PLACES_API_KEY:
        logger.warning("PLACES_API_KEY is not set. Cannot perform reverse geocoding.")
        return "Unknown location (API key missing)"
        
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"latlng": f"{lat},{lon}", "key": PLACES_API_KEY}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "OK" and data.get("results"):
                return data["results"][0]["formatted_address"]
            else:
                logger.warning(f"Reverse geocoding returned no results: {data.get('status')}")
                return "Unknown location"
        except httpx.RequestError as e:
            logger.error(f"Reverse geocoding request failed: {e}")
            return "Unknown location"

# --- Core WebSocket Logic ---
async def gemini_websocket_handler(request):
    """
    Handles the main WebSocket connection, integrating with the Gemini LiveConnect API.
    """
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    session_id = f"{int(asyncio.get_event_loop().time())}-{uuid.uuid4().hex[:8]}"
    logger.info(f"New client connection established. Assigning session ID: {session_id}")

    state = {
        "startLocation": None, "endLocation": None, "startDate": None,
        "startTime": None, "rideConfirmation": False, "authorization_token": None,
    }
    tools = types.Tool(function_declarations=[get_fare_details, book_ride])

    try:
        # 1. Authentication
        logger.info(f"[{session_id}] Waiting for authentication message.")
        auth_msg = await ws.receive(timeout=10.0)

        if auth_msg.type == aiohttp.WSMsgType.TEXT:
            auth_data = json.loads(auth_msg.data)
            if auth_data.get("type") == "auth" and auth_data.get("token"):
                state["authorization_token"] = auth_data["token"]
                state["user_name"] = auth_data.get("name", "Unknown")
                # This address is hardcoded in the original script.
                # If lat/lon were used, a call to reverse_geocode would go here.
                state["address"] = "Palm Jumeirah, Dubai"
                logger.info(f"[{session_id}] Authentication successful for user: {state['user_name']}.")
                await ws.send_json({"type": "auth_status", "status": "success", "session_id": session_id})
            else:
                logger.warning(f"[{session_id}] Invalid authentication data received. Closing.")
                await ws.send_json({"error": "Invalid authentication data.", "session_id": session_id})
                await ws.close()
                return ws
        else:
            logger.warning(f"[{session_id}] First message was not valid text. Closing connection.")
            await ws.send_json({"error": "Authentication required as first message.", "session_id": session_id})
            await ws.close()
            return ws

        # 2. Gemini LiveConnect Setup
        dubai_tz = pytz.timezone("Asia/Dubai")
        now_in_dubai = datetime.now(dubai_tz)
        SYSTEM_PROMPT = f"""You are Tala, an intelligent AI assistant based in the UAE. Your primary goal is assisting users with booking rides and location suggestions in the UAE.
Suggest location recommendations to the user based on your knowledge of the UAE. Always respond in English.

User Name: {state.get("user_name", "Unknown")}
Current User location: {state.get("address", "Unknown")}
Current UAE Date: {now_in_dubai.strftime("%d-%m-%Y")} (DD-MM-YYYY)
Current UAE Time: {now_in_dubai.strftime("%I:%M %p")} (H:MM AM/PM)
"""
        config = types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            input_audio_transcription={},
            output_audio_transcription={},
            system_instruction=types.Content(parts=[types.Part(text=SYSTEM_PROMPT)]),
            speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore"))),
            tools=[tools]
        )

        # 3. Establish Connection and Run Communication Tasks
        logger.info(f"[{session_id}] Connecting to Gemini LiveConnect API.")
        async with client.aio.live.connect(model=model_id, config=config) as session:
            logger.info(f"[{session_id}] Gemini session started. Beginning communication loop.")

            # TASK A: Gemini -> Client
            async def gemini_to_client_task():
                booking_confirmed = False
                async for gemini_message in session.receive():
                    if ws.closed: break
                    
                    # Audio Data
                    if gemini_message.data is not None:
                        pcm_data = base64.b64decode(gemini_message.data)
                        audio_chunk_b64 = base64.b64encode(pcm_data).decode('utf-8')
                        await ws.send_json({"type": "audio_chunk", "audio_chunk": audio_chunk_b64, "session_id": session_id})

                    # Transcription Chunks
                    if gemini_message.server_content:
                        if gemini_message.server_content.input_transcription:
                            await ws.send_json({"type": "chunk", "transcription_chunk": gemini_message.server_content.input_transcription.text, "session_id": session_id})
                        if gemini_message.server_content.output_transcription:
                            await ws.send_json({"type": "chunk", "response_chunk": gemini_message.server_content.output_transcription.text, "session_id": session_id})

                    # Tool Call Handling
                    if gemini_message.tool_call:
                        function_responses = []
                        for fc in gemini_message.tool_call.function_calls:
                            try:
                                n8n_payload = {"session_id": session_id, "state": state, "headers": {"authorization": state.get("authorization_token", "")}}
                                if fc.name == "get_fare_details":
                                    logger.info(f"[{session_id}] Tool call: get_fare_details with args: {fc.args}")
                                    state.update(fc.args)
                                    n8n_response = await call_n8n_webhook(n8n_payload)
                                    if n8n_response.get("fare"): state["fare"] = n8n_response.get("fare")
                                    function_responses.append(types.FunctionResponse(id=fc.id, name=fc.name, response=n8n_response))
                                elif fc.name == "book_ride":
                                    logger.info(f"[{session_id}] Tool call: book_ride with args: {fc.args}")
                                    state.update(fc.args)
                                    n8n_response = await call_n8n_webhook(n8n_payload)
                                    if n8n_response.get("status") == "BOOKING_CONFIRMED": booking_confirmed = True
                                    function_responses.append(types.FunctionResponse(id=fc.id, name=fc.name, response=n8n_response))
                            except Exception as e:
                                logger.error(f"[{session_id}] Error processing tool call '{fc.name}': {e}")
                                function_responses.append(types.FunctionResponse(id=fc.id, name=fc.name, response={"error": str(e)}))
                        
                        await session.send_tool_response(function_responses=function_responses)

                    # Turn Completion
                    if gemini_message.server_content and gemini_message.server_content.turn_complete:
                        logger.info(f"[{session_id}] Gemini turn complete.")
                        await asyncio.sleep(0.5) # Small delay to ensure final chunks are sent
                        await ws.send_json({"type": "final", "session_id": session_id})
                        if booking_confirmed:
                            logger.info(f"[{session_id}] Sending booking confirmation to client.")
                            await ws.send_json({"type": "confirm"})
                            booking_confirmed = False # Reset for next turn
                        break # Exit this inner loop to wait for next client input

            # TASK B: Client -> Gemini
            async def client_to_gemini_task():
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        if data.get("audio"):
                            audio_bytes = base64.b64decode(data["audio"])
                            await session.send_realtime_input(audio=types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000"))
                        elif data.get("text"):
                             logger.info(f"[{session_id}] Received text from client: {data['text']}")
                             await session.send_client_content(turns={"role": "user", "parts": [{"text": data['text']}]}, turn_complete=True)
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        logger.info(f"[{session_id}] Client disconnected.")
                        break
            
            # Run tasks concurrently
            await asyncio.gather(client_to_gemini_task(), gemini_to_client_task())

    except asyncio.CancelledError:
        logger.info(f"[{session_id}] Connection cancelled, likely client disconnect.")
    except Exception as e:
        logger.error(f"[{session_id}] An unexpected error occurred in the websocket handler: {e}", exc_info=True)
        # Try to send an error message if the socket is still open
        if not ws.closed:
            await ws.send_json({"error": str(e), "session_id": session_id})
    finally:
        logger.info(f"[{session_id}] Closing connection.")
        await ws.close()

    return ws

# --- HTTP Health Check Handler ---
async def health_check(request):
    """
    Simple HTTP endpoint for health checks by deployment services like Render.
    """
    return web.Response(status=200, text="OK")

# --- Application Setup and Server Start ---
if __name__ == "__main__":
    app = web.Application()
    
    # Add routes for websocket and health checks
    app.add_routes([
        web.get('/ws', gemini_websocket_handler),
        web.get('/', health_check),
        web.get('/health', health_check)
    ])

    # Get port from environment variable, default to 8080 for Render compatibility
    port = int(os.getenv("PORT", 8080))
    
    logger.info(f"Starting server on host 0.0.0.0 and port {port}")
    web.run_app(app, host="0.0.0.0", port=port)
