import asyncio
import websockets
import json
import requests
from google import genai
from google.genai.types import Content, LiveConnectConfig, Modality, Part

# Initialize Gemini Live API client
client = genai.Client(http_options={"api_version": "v1beta1"})
model_id = "gemini-2.0-flash-live-preview-04-09"
API_KEY = "YOUR_GOOGLE_API_KEY"
N8N_WEBHOOK_URL = "http://your-n8n-instance:5678/webhook/62675816-efda-4a15-8f79-3cc160015a44"  # From Webhook1

# System prompt to ensure structured output
SYSTEM_PROMPT = """
You are a ride-booking assistant in the UAE (UTC+4). Your job is to guide users conversationally to book a ride. For each user input, respond with a JSON object in the format:
{
  "response": "string", // Conversational response for the user
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

async def call_n8n_webhook(data):
    """Send structured output to n8n webhook"""
    headers = {"Content-Type": "application/json"}
    response = requests.post(N8N_WEBHOOK_URL, json=data, headers=headers)
    return response.json()

async def handle_websocket(websocket, path):
    """Handle WebSocket connection from frontend"""
    session_id = f"{int(asyncio.get_event_loop().time())}-{int(hash(path)) % 100000}"
    state = {
        "startLocation": None,
        "endLocation": None,
        "startDate": None,
        "startTime": None,
        "selectedSlot": None,
        "rideConfirmation": False,
        "rideRejection": False
    }

    async with client.aio.live.connect(
        model=model_id,
        config=LiveConnectConfig(response_modalities=[Modality.TEXT])
    ) as session:
        # Send system prompt
        await session.send_client_content(
            turns=Content(role="system", parts=[Part(text=SYSTEM_PROMPT)])
        )

        async for message in websocket:
            try:
                # Assume message is JSON with { "text": string, "audio": base64, "session_id": string }
                data = json.loads(message)
                user_input = data.get("text") or data.get("audio")  # Handle text or audio
                if not user_input:
                    await websocket.send(json.dumps({"error": "No input received", "session_id": session_id}))
                    continue

                # Send user input to Gemini
                await session.send_client_content(
                    turns=Content(role="user", parts=[Part(text=user_input)])
                )

                # Process Gemini response
                response_text = []
                for gemini_message in await session.receive():
                    if gemini_message.text:
                        response_text.append(gemini_message.text)

                # Parse Gemini's response (expected to be JSON)
                try:
                    gemini_output = json.loads("".join(response_text))
                    response = gemini_output.get("response", "Sorry, something went wrong.")
                    state = gemini_output.get("state", state)
                except json.JSONDecodeError:
                    response = "Sorry, something went wrong. Please try again."
                    state = state

                # Prepare payload for n8n
                n8n_payload = {
                    "body": {
                        "message": user_input,
                        "session_id": data.get("session_id", session_id),
                        "response": response,
                        "state": state
                    },
                    "headers": {"authorization": data.get("authorization", "")}
                }

                # Send to n8n
                n8n_response = await call_n8n_webhook(n8n_payload)

                # Send response to frontend
                frontend_response = {
                    "response": response,
                    "session_id": session_id,
                    "state": state
                }
                await websocket.send(json.dumps(frontend_response))
            except Exception as e:
                error_response = {"error": str(e), "session_id": session_id}
                await websocket.send(json.dumps(error_response))

# Start WebSocket server
start_server = websockets.serve(handle_websocket, "0.0.0.0", 8765)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()