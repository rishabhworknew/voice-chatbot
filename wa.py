import json
import requests
from datetime import datetime
from google.genai import types
from google import genai
from urllib.parse import quote
import httpx
import logging
import asyncio
conversation_history = {}

g = genai.Client(api_key="AIzaSyAkDBXjvsx9SfuROqlymWfyjbvbFr3MTgM")



USER_PHONE   = ""
API_ENDPOINT = "http://0.0.0.0:5000/get_details"

async def call_n8n_webhook(ride_data):
    N8N_WEBHOOK_URL = "https://n8n.noorride.com/webhook/chatbot"
    if not N8N_WEBHOOK_URL:
        logger.error("N8N_WEBHOOK_URL is not set.")
        return {"error": "Webhook URL is not configured."}
    
    # Format the data with session_id and state object
    payload = {
        "session_id": str(USER_PHONE),  # Using phone number as session ID
        "state": {
            "startLocation": ride_data.get("startLocation", ""),
            "endLocation": ride_data.get("endLocation", ""),
            "startDate": ride_data.get("startDate", ""),
            "startTime": ride_data.get("startTime", ""),
            "rideConfirmation": ride_data.get("rideConfirmation", False)
        },
        "headers": {
            "authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI2Njg2MDc4MjAyNWMwYjcyNThiZGY4MzkiLCJpYXQiOjE3NTE2MjAwNTcsImV4cCI6MTc1NDIxMjA1Nywicm9sZSI6InVzZXIiLCJ0eXBlIjoiYWNjZXNzIn0.GsJnb1bxPxLAzBmWqrKVxIOPCcyYnDr1Lf411zOJ6Po"
        }
    }
    
    headers = {"Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(N8N_WEBHOOK_URL, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
    except httpx.RequestError as e:
        logger.error(f"Error calling n8n webhook: {e}")
        return {"error": f"Failed to connect to webhook service: {e}"}

def get_user_rides() -> dict:
    """
    Fetch all ride bookings for the current user.
    Leading '+' in the phone number is percent-encoded as '%2B'.
    """
    e164 = USER_PHONE if USER_PHONE.startswith("+") else f"+{USER_PHONE}"
    encoded = quote(e164, safe="")
    resp = requests.get(f"{API_ENDPOINT}?phonenumber={encoded}")
    resp.raise_for_status()
    return resp.json()

# ——————————————————————————
# 2) Create your Gemini model
# ——————————————————————————

get_fare_details = {
    "name": "get_fare_details",
    "description": "Processes ride booking details to fetch service fare / time slots. All parameters to be sent in English.",
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
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
# Create a Tool object with the function declarations
tools = [types.Tool(
    function_declarations=[get_fare_details, book_ride]
)]

config = types.GenerateContentConfig(
    system_instruction=("You are the official WhatsApp chatbot for Noor Ride. "
                       "To get a user's ride history, automatically call get_user_rides(). "
                       "If user wants to book a ride call get_fare_details. "
                       "If the user confirms the fare, call book_ride."),
    tools=tools)

model = g.models.generate_content(
    model="gemini-2.5-flash",
    # generation_config=generation_config,
    config=config,
    contents="i want to go from gems school to dubai airport at 9 pm on 17/07/2025 "
)
authorization_token = "Bearer fdvmQsDfO0aXp3Tf1PjqIk:APA91bFAbBXr2VFPP2Ze5IXhp9I5HI24NDgx_l5NlxgWO1U2KZre0bFlU1s6YM3NF5Rtnc1cObjb9ZfQT4gOPXggqs6O8Z3DrmODPEAqW_xQas_68OJoDXE"
session_id = "123456789"
if model.candidates[0].content.parts[0].function_call:
    function_call = model.candidates[0].content.parts[0].function_call
    print(f"Function to call: {function_call.name}")
    print(f"Arguments: {function_call.args}")
    #  In a real app, you would call your function here:
    #  result = schedule_meeting(**function_call.args)

    if function_call.name == "get_fare_details":
        async def process_webhook():
            # Include all the necessary ride details in the payload
            n8n_payload = {
                "session_id": session_id,
                "headers": {
                    "authorization": authorization_token,
                    "Content-Type": "application/json"
                },
                "startLocation": function_call.args.get("startLocation", ""),
                "endLocation": function_call.args.get("endLocation", ""),
                "startDate": function_call.args.get("startDate", ""),
                "startTime": function_call.args.get("startTime", "")
            }
            return await call_n8n_webhook(n8n_payload)
            
        try:
            n8n_response = asyncio.run(process_webhook())
            print(f"Webhook response: {n8n_response}")
            # function_responses.append(types.FunctionResponse(id=fc.id, name=fc.name, response=n8n_response))
        except Exception as e:
            print(f"Error processing webhook: {str(e)}")
            n8n_response = {"error": f"Failed to get fare details: {str(e)}"}

else:
    print("No function call found in the response.")
    print(model.text)



import sqlite3
from datetime import datetime, timedelta

# Get a connection to the SQLite database
def get_connection():
    return sqlite3.connect("threads_db.sqlite")

# Ensure the database and table exist
def setup_database():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS threads (
                wa_id TEXT PRIMARY KEY,
                history TEXT,
                current_state TEXT,
                timestamp DATETIME
            )
        """)
        conn.commit()

# Call setup_database() at the start of your application
setup_database()


import json
from datetime import datetime

def store_thread(wa_id, history, current_state):
    print("Storing thread...")
    current_time = datetime.now().isoformat()  # Store timestamp as ISO format string

    # Serialize the `history` list to a JSON string
    history_json = json.dumps(history)

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO threads (wa_id, history, current_state, timestamp)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(wa_id) DO UPDATE SET
                history = excluded.history,
                current_state = excluded.current_state,
                timestamp = excluded.timestamp
        """, (wa_id, history_json, current_state, current_time))
        conn.commit()
        print(f"Thread for wa_id '{wa_id}' stored successfully.")



def check_if_thread_exists(wa_id):
    print("Checking thread...")
    current_time = datetime.now()

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT history, current_state, timestamp FROM threads WHERE wa_id = ?", (wa_id,))
        row = cursor.fetchone()

        if row:
            history_json, current_state, thread_timestamp = row
            thread_timestamp = datetime.fromisoformat(thread_timestamp)

            # Deserialize `history` JSON string to a Python list
            history = json.loads(history_json)

            if current_time - thread_timestamp > timedelta(minutes=5):
                print(f"Thread for wa_id {wa_id} is older than 5 minutes. Deleting...")
                delete_thread_history(wa_id)
                return None, None
            else:
                print(f"Thread for wa_id {wa_id} is within 5 minutes. Returning history and state.")
                return history, current_state

        print(f"No thread found for wa_id {wa_id}.")
        return None, None

def delete_thread_history(wa_id):
    print("Deleting thread...")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM threads WHERE wa_id = ?", (wa_id,))
        conn.commit()
        print(f"Thread for wa_id '{wa_id}' deleted successfully.")



def generate_response(message_body, wa_id, name,message_type,prev):
    print(prev)
    global current_state
    print(message_type)
    print(message_body)
    print(type(message_body))
    print(current_state)
    print(current_state)
    global USER_PHONE
    USER_PHONE = "971544027717"
    list = ["hello","hii","HI","HELLO","hi","Hello","Hi","Hii"]
    # Extract conversation state from message_body
    message_body_lower = message_body.lower()
    if message_body_lower in list:
        delete_thread_history(wa_id)
        send_wati_sms_greet(wa_id,name)
        return ""
    # Check if there is already a conversation history for the wa_id
    history , current_state = check_if_thread_exists(wa_id)
    
    # If no history exists, initialize a new conversation
    if history is None :
        history = []

    # Add the new user message to the history
    history.append({
        "role": "user",
        "parts": [message_body]
    })
    chat = model.start_chat(
        history=history,
        enable_automatic_function_calling=True
    )
    # Start or continue the chat session
    

    '''if len(history) > 2:
        j = chat_session.history[-2].role
        y = chat_session.history[-2].parts
        y = str(y)
        desired_string = y.strip('[text: "').strip('" ]')
        if j == "model" and "confirm" in desired_string and current_state == "booking":
            current_state = "confirm"'''

    # Get the model's response
    
    if "enquiry" in message_body_lower:
        current_state = "enquiry"
    elif "booking" in message_body_lower:
        current_state = "enquiry"
    elif "complains and feedbacks" in message_body_lower:
        current_state = "enquiry"
    elif "confirm" in message_body_lower:
        current_state = "enquiry"
    if current_state is None:
        response = chat.send_message(message_body)

        text_response = response.text or f"<Invoked {getattr(response, 'tool_invocation', type('dummy', (), {'name': 'unknown'})) .name}>"

        # Append assistant message properly
        history.append({
            "role": "model",  # ✅ Gemini expects "model", not "assistant"
            "parts": [{"text": text_response}]
        })

        # Optional: If you're tracking function calls separately
        if hasattr(response, "function_call") and response.function_call:
            function_call_text = f"<Called {response.function_call.name}>"
            print(f"<Called {response.function_call.name}>")
            print(f"<Called {response.function_call.args}>")
            history.append({
                "role": "model",
                "parts": [{"text": function_call_text}]
            })

        store_thread(wa_id, history, current_state)
        return response.text

        
        