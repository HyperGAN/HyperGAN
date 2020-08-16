import asyncio
import json
import websockets

class WebsocketServer:
    def __init__(self):
        port = 9999
        for i in range(100):
            try:
                asyncio.get_event_loop().run_until_complete(self.serve(port))
                break;
            except OSError:
                port += 1
                continue
            print(f"Starting on port {port}")
        asyncio.get_event_loop().run_forever()

    async def listen(self, websocket, path):
        routes = self.routes()
        print("Found routes")
        while True:
            message = await websocket.recv()
            request = json.loads(message)
            await routes[request["action"]](websocket, request)

    async def connect(self, websocket, request):
        await websocket.send(json.dumps({"action": "connect"}))

    def routes(self):
        return {
            "connect" : self.connect,
            "get_samples" : self.get_samples
        }

    async def get_samples(self, websocket, request):
        sample = gan.sample()
        await websocket.send(json.dumps({"action": "samples_start"}))
        [await websocket.sendall(image) for image in sample.to_images(format='png')]
        await websocket.send(json.dumps({"action": "samples_end"}))

    def serve(self, port):
        return websockets.serve(self.listen, "localhost", port)
