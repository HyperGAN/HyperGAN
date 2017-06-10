import hyperchamber as hc
import os

class Configuration:
    def find(configuration):
        dirname = os.path.dirname(os.path.realpath(__file__))
        paths = [dirname + "/configurations/", '~/.hypergan/configs/']
        for path in paths:
            file_path = path + configuration
            if os.path.exists(file_path):
                return file_path
    def load(configuration):
        return hc.Selector().load(Configuration.find(configuration))
    def default():
        return Configuration.load('default.json')
