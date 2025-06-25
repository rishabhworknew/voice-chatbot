import asyncio
import os
from aiohttp import web
import aiohttp
import websockets

# WebSocket handler for echoing messages
async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                print(f"Received: {msg.data}")
                await ws.send_str(f"Echo: {msg.data}")
            elif msg.type == aiohttp.WSMsgType.CLOSED:
                print("Client disconnected")
                break
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await ws.close()

    return ws

# HTTP handler for health checks (HEAD requests)
async def health_check(request):
    if request.method == "HEAD" or request.method == "GET":
        return web.Response(status=200, text="OK")
    return web.Response(status=405, text="Method Not Allowed")

# Create the aiohttp application
app = web.Application()
app.add_routes([web.get('/ws', websocket_handler),
                 web.head('/', health_check),
                 web.get('/', health_check)])

# Start the server
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    web.run_app(app, host="0.0.0.0", port=port)