import asyncio
import websockets
import json
from dotenv import load_dotenv
import os
from datetime import datetime
from google import genai
from google.genai import types
from google.genai.types import Tool
import uuid
import logging
import base64
import pytz
import httpx
from http.server import BaseHTTPRequestHandler, HTTPServer # For a simple HTTP server

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
PLACES_API_KEY = os.getenv("PLACES_API_KEY")
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is missing!")

client = genai.Client(api_key=API_KEY)
model_id = "gemini-2.0-flash-live-001"

get_fare_details = {
    "name": "get_fare_details",
    "description": "Processes ride booking details to fetch service fare / time slots.All parameters to be send in english.",
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

async def call_n8n_webhook(data):
    """Send structured output to n8n webhook"""
    headers = {"Content-Type": "application/json"}
    async with httpx.AsyncClient() as client:
        response = await client.post(N8N_WEBHOOK_URL, json=data, headers=headers)
        response.raise_for_status()
        return response.json()

async def reverse_geocode(lat, lon):
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "latlng": f"{lat},{lon}",
        "key": PLACES_API_KEY
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data["status"] == "OK" and data["results"]:
                return data["results"][0]["formatted_address"]
            else:
                logger.warning(f"Google returned no results: {data}")
                return "Unknown location"
        else:
            logger.warning(f"Geocoding failed with HTTP {response.status_code}")
            return "Unknown location"

async def handle_websocket(websocket):
    dubai_tz = pytz.timezone("Asia/Dubai")
    now_in_dubai = datetime.now(dubai_tz)
    current_dubai_time = now_in_dubai.strftime("%I:%M %p")
    current_dubai_date = now_in_dubai.strftime("%d-%m-%Y")
    logger.info(f"Current Dubai time: {current_dubai_time}, date: {current_dubai_date}")



    session_id = f"{int(asyncio.get_event_loop().time())}-{uuid.uuid4().hex[:8]}"
    state = {
        "startLocation": None,
        "endLocation": None,
        "startDate": None,
        "startTime": None,
        "rideConfirmation": False,
        "authorization_token": None,
    }
    tools = types.Tool(function_declarations=[get_fare_details, book_ride])

    logger.info("New client connection established.")


    try:
        auth_message = await websocket.recv()
        auth_data = json.loads(auth_message)

        if auth_data.get("type") == "auth" and auth_data.get("token"):
            state["authorization_token"] = auth_data["token"]
            state["user_name"] = auth_data.get("name", "Unknown")
            state["latitude"] = auth_data.get("latitude", "Unknown")
            state["longitude"] = auth_data.get("longitude", "Unknown")
            if state["latitude"] != "Unknown" and state["longitude"] != "Unknown":
                state["address"] = "Palm Jumeirah, Dubai"
            else:
                state["address"] = "Unknown location"
            logger.info("########################")
            logger.info(state)
            logger.info("########################")
            await websocket.send(json.dumps({"type": "auth_status", "status": "success", "session_id": session_id}))
        else:
            logger.warning("First message was not a valid authentication message. Closing connection.")
            await websocket.send(json.dumps({"error": "Authentication required as first message.", "session_id": session_id}))
            await websocket.close()
            return

        SYSTEM_PROMPT = f"""You are Tala, an intelligent AI assistant based in the UAE . Your primary goal is assisting users with booking rides and location suggestions in the UAE .
Suggest location recommendations to the user based on the your knowledge of the UAE.
Always respond in English . 

User Name: {state.get("user_name", "Unknown")}
Current User location: {state.get("address", "Unknown")}
Current UAE Date: {current_dubai_date} , DD-MM-YYYY format
Current UAE Time: {current_dubai_time} , H:MM AM/PM format


**THE GOLDEN RULES - NON-NEGOTIABLE**

1.  **NEVER MAKE UP A FARE.** The ride fare is dynamic and unpredictable. The fare is completely unknown to you until the `get_fare_details` function returns it. Stating a fare you were not given by the function is a critical failure.
2.  **ALWAYS USE YOUR TOOLS.** Your only job in ride booking is to collect information and then call the functions in the correct order. Do not try to complete the booking process on your own.

### RIDE BOOKING WORKFLOW ---

**Step 1: Information Gathering**

Your task is to collect these four pieces of information one by one in a natural way adapting to the conversation:
* `startLocation` (Where the ride begins)
* `endLocation` (Where the ride ends)
* `startTime` (The desired pickup time)
* `startDate` (The desired pickup date)

**Step 2: Processing Ride Details & Getting the Fare**

* **TRIGGER:** As soon as you have the four pieces of information (`startLocation`, `endLocation`, `startTime`, `startDate`), you MUST immediately stop the conversation and call the `get_fare_details` function. This is your only next action.
* **FUNCTION PURPOSE:** This function checks vehicle availability and calculates the official fare.

**Handling Function Responses:**

* **Success:** If the function returns a fare, present the fare to the user and ask for confirmation. 
* **Unserviceable Location:** If a location is invalid, relay this to the user and ask for a corrected location. Then, you must call the `get_fare_details` function again with the new information.
* **Alternative Time:** If your time is unavailable , the function returns the closest available times, relay this to the user and ask for a new time. Then, you must call the `get_fare_details` function again with the new information.

**Critical Rules for Processing:**

* **Always Call the Function:** Call `get_fare_details` every time you have the four required details, even if the user changes just one piece of information (like the time or location).
* **The Function is the Only Source of Truth:** Only present the exact fare returned by this function.

**Step 3: Booking Confirmation**

* **TRIGGER:** You can ONLY call the `book_ride` function AFTER you have presented the fare from `get_fare_details` and the user has given a clear, affirmative confirmation (e.g., "Yes," "Book it," "Confirm").
"""
        config = types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            input_audio_transcription={},
            output_audio_transcription={},
            system_instruction=types.Content(parts=[types.Part(text=SYSTEM_PROMPT)]),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Kore"
                    )
                )
            ),
            tools=[tools] 
        )



        async with client.aio.live.connect(model=model_id, config=config) as session:
            async def gemini_to_client():
                try:
                    booking_confirmed = False
                    while True:
                        async for gemini_message in session.receive():
                            if gemini_message.data is not None:
                                pcm_data = base64.b64decode(gemini_message.data)
                                audio_chunk_b64 = base64.b64encode(pcm_data).decode('utf-8')
                                await websocket.send(json.dumps({
                                    "type": "audio_chunk", "audio_chunk": audio_chunk_b64, "session_id": session_id
                                }))

                            if gemini_message.server_content and gemini_message.server_content.input_transcription:
                                await websocket.send(json.dumps({
                                    "type": "chunk", "transcription_chunk": gemini_message.server_content.input_transcription.text, "session_id": session_id
                                }))
                                print(" transcription:", gemini_message.server_content.input_transcription.text)
                            if gemini_message.server_content and gemini_message.server_content.output_transcription:
                                await websocket.send(json.dumps({
                                    "type": "chunk", "response_chunk": gemini_message.server_content.output_transcription.text, "session_id": session_id
                                }))
                                print(" response:", gemini_message.server_content.output_transcription.text)

                            if gemini_message.tool_call:
                                print(" tool call:", gemini_message.tool_call)
                                function_responses = []
                                for fc in gemini_message.tool_call.function_calls:
                                    try:
                                        if fc.name == "get_fare_details":
                                            print(" get_fare_details:", fc.args)
                                            params = fc.args
                                            state.update(params)
                                            try:
                                                if state.get("startDate"): datetime.strptime(state["startDate"], "%d-%m-%Y")
                                                if state.get("startTime"): datetime.strptime(state["startTime"], "%I:%M %p")
                                            except ValueError as e:
                                                logger.error(f"Invalid state format: {str(e)}")
                                                function_responses.append(types.FunctionResponse(id=fc.id, name=fc.name, response={"error": f"Invalid date/time format: {str(e)}"}))
                                                continue

                                            n8n_payload = {"session_id": session_id, "state": state, "headers": {"authorization": state.get("authorization_token", "")}}
                                            n8n_response = await call_n8n_webhook(n8n_payload)
                                            fare = n8n_response.get("fare")
                                            if fare:
                                                print(" Fare returned by n8n:", fare)
                                                state["fare"] = fare
                                            if "state" in n8n_response:
                                                state.update(n8n_response["state"])
                                            print(" n8n_response:", n8n_response)
                                            function_responses.append(types.FunctionResponse(
                                                id=fc.id,
                                                name=fc.name,
                                                response=n8n_response
                                            ))
                                        elif fc.name == "book_ride":
                                            params = fc.args
                                            state.update(params)
                                            n8n_payload = {"session_id": session_id, "state": state, "headers": {"authorization": state.get("authorization_token", "")}}
                                            n8n_response = await call_n8n_webhook(n8n_payload)
                                            if n8n_response.get("status") == "BOOKING_CONFIRMED":
                                                booking_confirmed = True

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
                                await asyncio.sleep(0.5)
                                await websocket.send(json.dumps({"type": "final", "session_id": session_id}))
                                print("Gemini turn complete.")
                                if booking_confirmed:
                                    try:
                                        await websocket.send(json.dumps({
                                            "type": "confirm"
                                        }))
                                    except websockets.exceptions.ConnectionClosed:
                                        logger.info("WebSocket closed before sending booking confirmation.")
                                    booking_confirmed = False
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
                            await session.send_client_content(turns={"role": "user", "parts": [{"text": user_input}]}, turn_complete=True)
                            print("Client sent text to gemini :", user_input)

                        elif audio_input:
                            audio_bytes = base64.b64decode(audio_input)
                            await session.send_realtime_input(
                                audio=types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000")
                            )
                            # print("Client sent audio to gemini :", audio_input)

                        """elif data.get("action") == "audio_input_ended": # New condition
                            await session.send_realtime_input(audio_stream_end=True)
                            print("Client signaled end of audio input for the turn to Gemini.")"""

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

# Simple HTTP handler for health checks
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_error(404)

async def run_health_check_server():
    health_port = int(os.getenv("HEALTH_CHECK_PORT", 8080)) # A different port for health checks
    httpd = HTTPServer(("0.0.0.0", health_port), HealthCheckHandler)
    logger.info(f"Health check server started on http://0.0.0.0:{health_port}")
    await asyncio.to_thread(httpd.serve_forever)


async def main():
    websocket_port = int(os.getenv("PORT", 10000))
    # Run both the WebSocket server and the HTTP health check server concurrently
    await asyncio.gather(
        websockets.serve(handle_websocket, "0.0.0.0", websocket_port),
        run_health_check_server()
    )
    print(f"WebSocket server started on ws://0.0.0.0:{websocket_port}")
    await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())