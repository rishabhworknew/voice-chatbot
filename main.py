import asyncio
import websockets
import os

# WebSocket handler that echoes messages
async def echo(websocket, path):
    try:
        async for message in websocket:
            print(f"Received: {message}")
            await websocket.send(f"Echo: {message}")
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

# Start the WebSocket server
async def main():
    # Get the port from the environment variable (Render sets this)
    port = int(os.getenv("PORT", 8000))
    # Start the server, binding to 0.0.0.0 for Render
    server = await websockets.serve(echo, "0.0.0.0", port)
    print(f"WebSocket server started on port {port}")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())