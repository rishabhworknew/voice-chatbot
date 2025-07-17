async def call_places_api(text_query: str):
    """Calls the Google Places API to find places."""
    headers = {
        'Content-Type': 'application/json',
        'X-Goog-Api-Key': PLACES_API_KEY,
        'X-Goog-FieldMask': 'places.displayName,places.formattedAddress',
    }
    data = {
        'textQuery': text_query,
        'maxResultCount': 5
    }
    url = 'https://places.googleapis.com/v1/places:searchText'

    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: requests.post(url, json=data, headers=headers))
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Error calling Places API: {e}")
        return {"error": f"Failed to connect to Places API: {str(e)}"}

search_places = {
    "name": "search_places",
    "description": "Searches for places like restaurants, cafes, or landmarks based on a text query.",
    "parameters": {
        "type": "object",
        "properties": {
            "textQuery": {
                "type": "string",
                "description": "The search query from the user, for example: 'italian restaurants in dubai' or 'cafes near dubai mall'."
            }
        },
        "required": ["textQuery"]
    }
}

search_places = {
    "name": "search_places",
    "description": "Searches for places like restaurants, cafes, or landmarks based on a text query.",
    "parameters": {
        "type": "object",
        "properties": {
            "textQuery": {
                "type": "string",
                "description": "The search query from the user, for example: 'italian restaurants in dubai' or 'cafes near dubai mall'."
            }
        },
        "required": ["textQuery"]
    }
}