import asyncio
import os
import requests
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "http://localhost:5678/webhook/chatbot")

# Define the function declaration for the model
process_ride_details = {
    "name": "process_ride_details",
    "description": "Processes ride booking details from the backend to fetch fare and available time slots closest to users start time for a ride in the UAE.",
    "parameters": {
        "type": "object",
        "properties": {
            "startLocation": {
                "type": "string",
                "description": "The starting location of the ride, e.g., 'Dubai Mall' or 'Dubai Airport'."
            },
            "endLocation": {
                "type": "string",
                "description": "The destination location of the ride, e.g., 'Dubai Airport' or 'Burj Khalifa'."
            },
            "startDate": {
                "type": "string",
                "description": "The date of the ride in DD-MM-YYYY format, e.g., '24-05-2025'."
            },
            "startTime": {
                "type": "string",
                "description": "The time of the ride in H:MM AM/PM format, e.g., '3:30 PM'."
            }
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

# Configure the client and tools
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
model = "gemini-2.0-flash-live-001"
tools = [{"function_declarations": [process_ride_details]}]
config = {"response_modalities": ["TEXT"], "tools": tools}

async def main():
    try:
        # Establish a live session
        async with client.aio.live.connect(model=model, config=config) as session:
            # Send the initial prompt
            prompt = "Book a ride from Dubai Mall to Dubai Airport at 9 PM on 01-06-2025."
            await session.send_client_content(turns={"parts": [{"text": prompt}]})

            # Process responses in real-time
            async for chunk in session.receive():
                if chunk.server_content:
                    if chunk.text is not None:
                        print(f"Server response: {chunk.text}")
                elif chunk.tool_call:
                    function_responses = []
                    for fc in chunk.tool_call.function_calls:
                        if fc.name == "process_ride_details":
                            # Process ride details using n8n webhook
                            args = fc.args
                            print(f"Function called: {fc.name} with arguments: {args}")

                            # Call n8n webhook with ride details
                            try:
                                n8n_response = await call_n8n_webhook(args)
                                # Create function response
                                function_response = types.FunctionResponse(
                                    id=fc.id,
                                    name=fc.name,
                                    response={"result": n8n_response}
                                )
                                function_responses.append(function_response)
                            except requests.RequestException as e:
                                print(f"Error calling n8n webhook: {str(e)}")
                    # Send the function response back to the session
                    await session.send_tool_response(function_responses=function_responses)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())