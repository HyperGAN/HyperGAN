import hyperchamber as hc
import os

class Configuration:
    def find(configuration):
        dirname = os.path.dirname(os.path.realpath(__file__))
        paths = [dirname + "/configurations/", os.path.abspath(os.path.expanduser('~/.hypergan/configs/'))+'/',
                 os.path.abspath(os.path.relpath("."))+"/" ]
        Configuration.paths = paths
        for path in paths:
            file_path = path + configuration
            file_path = os.path.realpath(file_path)
            if os.path.exists(file_path):
                return file_path
    def load(configuration):
        config_file = Configuration.find(configuration)
        if config_file is None:
            print("[hypergan] Could not find config named:", configuration, "checked paths", Configuration.paths)
        print("[hypergan] Loading config", config_file)
        return hc.Selector().load(config_file)
    def default():
        return Configuration.load('default.json')
