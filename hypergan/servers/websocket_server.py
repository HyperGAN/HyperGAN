import asyncio
import websockets

class WebsocketServer:
    def __init__(self):
        asyncio.get_event_loop().run_until_complete(self.serve())
        asyncio.get_event_loop().run_forever()

    def find_port(self):
        return 9999

    async def listen(self, websocket, path):
        while True:
            message = await websocket.recv()
            print(f"< {message}")
            await websocket.send("pong")
            print(f"< pong")

    def serve(self):
        print(f"Starting on port {self.find_port()}")
        return websockets.serve(self.listen, "localhost", self.find_port())
