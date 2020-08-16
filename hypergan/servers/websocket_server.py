import asyncio
import pathlib
import json
import websockets

class WebsocketServer:
    def __init__(self, gan):
        self.gan = gan
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
        #sample = self.gan.sample()
        #sample_images = sample.to_images(format='png')
        sample_images = []
        path = pathlib.Path(__file__).parent.parent.parent.joinpath("samples/default/000001.png").absolute()
        with open(path, 'rb') as f:
            sample_images.append(f.read())

        await websocket.send(json.dumps({"action": "samples_start"}))
        for image in sample_images:
            await websocket.send(json.dumps({"action": "sample_start"}))
            await websocket.send(image)
            await websocket.send(json.dumps({"action": "sample_end"}))
        await websocket.send(json.dumps({"action": "samples_end"}))

    def serve(self, port):
        return websockets.serve(self.listen, "localhost", port)
