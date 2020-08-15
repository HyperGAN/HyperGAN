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

    def check_if_ui_running(self):
        if not pathlib.Path(self.get_ui_tmp_file()).is_file():
            return False
        with open(self.get_ui_tmp_file(), 'r') as temp:
            pid = temp.read()
            try:
                if psutil.pid_exists(int(pid)):
                    return True
                else:
                    return False
            except ValueError:
                return False

    def get_ui_tmp_file(self):
        path = pathlib.Path(tempfile.gettempdir()).joinpath("hypergan.pid").absolute()
        return path

    def start_websocket_server(self):
        WebsocketServer()

    def spawn_ui(self):
        electron_path = which("electron")
        if electron_path is None:
            print("Could not find electron. Please make sure that it is available with `npm i -g electron`")

        if self.check_if_ui_running():
            print("Hypergan UI is already running.")
            return

        ui_path = pathlib.Path(__file__).parent.parent.joinpath("ui").absolute()
        print("Starting electron app")
        popen = subprocess.Popen([electron_path, ui_path], shell=False)
        with open(self.get_ui_tmp_file(), 'w') as temp:
            temp.write(str(popen.pid))
        self.ui_process = popen
        print("Electron app started")

    def spawn_websocket_server(self):
        print("Starting websocket server")
        #self.wss_process = torch.multiprocessing.spawn(self.start_websocket_server, join=False)
        print("Websocket server started")
