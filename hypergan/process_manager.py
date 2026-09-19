import pathlib
import psutil
import subprocess
import tempfile
import torch
from shutil import which

class ProcessManager:
    def __init__(self):
        self.ui_process = None
        self.wss_process = None

    def start_websocket_server(self):
        WebsocketServer()

    def spawn_websocket_server(self):
        print("Starting websocket server")
        #self.wss_process = torch.multiprocessing.spawn(self.start_websocket_server, join=False)
        print("Websocket server started")
