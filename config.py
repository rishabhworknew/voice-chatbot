import os
import logging
import httpx
from dotenv import load_dotenv

# --- Basic Setup ---
load_dotenv()
logger = logging.getLogger(__name__)

# --- Environment Variables ---
PLACES_API_KEY = os.getenv("PLACES_API_KEY")
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")


def get_system_prompt(state, current_dubai_date, current_dubai_time):
    return f"""You are Tala, an AI assistant based in the UAE. Your primary goal is assisting users with booking rides and location suggestions in the UAE.
Always respond in English.

User Name: {state.get("user_name", "Unknown")}
Current User location: {state.get("address", "Unknown")}
Current UAE Date: {current_dubai_date} , DD-MM-YYYY format
Current UAE Time: {current_dubai_time} , H:MM AM/PM format


**THE GOLDEN RULES - NON-NEGOTIABLE**

1. NEVER MAKE UP A FARE. The ride fare is dynamic and unpredictable. The fare is completely unknown to you until the `get_fare_details` function returns it. Stating a fare you were not given by the function is a critical failure.
2. ALWAYS USE YOUR TOOLS. Your only job in ride booking is to collect information and then call the functions in the correct order. Do not try to complete the booking process on your own.

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
* **Alternative Time:** If your time is unavailable, the function returns the closest available times, relay this to the user and ask for a new time. Then, you must call the `get_fare_details` function again with the new information.

**Critical Rules for Processing:**

* **Always Call the Function:** Call `get_fare_details` every time you have the four required details, even if the user changes just one piece of information (like the time or location).
* **The Function is the Only Source of Truth:** Only present the exact fare returned by this function.

**Step 3: Booking Confirmation**

* **TRIGGER:** You can ONLY call the `book_ride` function AFTER you have presented the fare from `get_fare_details` and the user has given a clear, affirmative confirmation (e.g., "Yes," "Book it," "Confirm").
"""

# --- Tool Definitions ---

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


# --- Utility Functions ---

async def call_n8n_webhook(data):
    if not N8N_WEBHOOK_URL:
        logger.error("N8N_WEBHOOK_URL is not set.")
        return {"error": "Webhook URL is not configured."}
        
    headers = {"Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(N8N_WEBHOOK_URL, json=data, headers=headers)
            response.raise_for_status()
            return response.json()
    except httpx.RequestError as e:
        logger.error(f"Error calling n8n webhook: {e}")
        return {"error": f"Failed to connect to webhook service: {e}"}


async def reverse_geocode(lat, lon):
    if not PLACES_API_KEY:
        logger.warning("PLACES_API_KEY is not set. Cannot perform reverse geocoding.")
        return "Unknown location (API key missing)"

    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "latlng": f"{lat},{lon}",
        "key": PLACES_API_KEY
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data["status"] == "OK" and data["results"]:
                    return data["results"][0]["formatted_address"]
                else:
                    logger.warning(f"Google returned no results for geocoding: {data}")
                    return "Unknown location"
            else:
                logger.warning(f"Geocoding failed with HTTP {response.status_code}")
                return "Unknown location"
    except httpx.RequestError as e:
        logger.error(f"Error during reverse geocoding request: {e}")
        return "Unknown location (request failed)"
