import hyperchamber as hc

class SkipConnections:
    def __init__(self):
        self.connections = {}

    def get(self, name, shape=None):
        conns = hc.Config(self.connections)
        conns = conns.name or []
        for con in conns:
            if con[0] == shape:
                return con[1]
        return None

    def set(self, name, value, shape = None):
        if not hasattr(self.connections, name):
            self.connections[name] = []
        self.connections[name].append([shape, value])
