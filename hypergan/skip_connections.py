import hyperchamber as hc
import tensorflow as tf

class SkipConnections:
    """
    Skip connections allow for cross-graph connections by shape.

    For example:

    ```python
        skip_connections.set('layer_filter', net) # net is 64x64x3
        skip_connections.set('layer_filter', net2) # net2 is 128x128x3
        skip_connections.get('layer_filter', [128, 128, 3]) #returns net2
        skip_connections.get('layer_filter', [64, 64, 3]) #returns net
    ```
    """
    def __init__(self):
        self.connections = {}

    def get(self, name, shape=None):
        if shape:
            shape = [int(x) for x in shape]
        
        connections = hc.Config(self.connections)
        if name in connections:
            conns = connections[name]
        else:
            conns = []
        for con in conns:
            if con[0] == shape:
                return con[1]
        return None

    def get_closest(self, name, shape=None):
        if shape:
            shape = [int(x) for x in shape]
        
        connections = hc.Config(self.connections)
        if name in connections:
            conns = connections[name]
        else:
            conns = []
        for con in conns:
            s1 = con[0]
            s2 = shape
            print("---->", s1, s2)
            if s1[1] >= s2[1]: 
                return con[1]
        return None


    def get_shapes(self, name):
        if name not in self.connections.keys():
            return None
        shapes = []
        if name not in self.connections.keys():
            return []
        for conn in self.connections[name]:
            if conn[0] not in shapes:
                shapes.append(conn[0])
        return shapes

    def clear(self, name, shape=None):
        new_connections = []
        for conn in self.connections[name]:
            if(shape == conn[0] or shape is None):
                pass
            else:
                new_connections.append(conn)
        self.connections[name] = new_connections

    def get_array(self, name, shape=None):
        if shape:
            shape = [int(x) for x in shape]
        connections = hc.Config(self.connections)
        if name in connections:
            conns = connections[name]
        else:
            conns = []
        return [con[1] for con in conns if shape is None or con[0] == shape]


    def set(self, name, value):
        shape = value.get_shape()
        if name not in self.connections:
            self.connections[name] = []
        self.connections[name].append([[int(x) for x in shape], value])
