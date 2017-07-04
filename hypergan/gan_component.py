import hyperchamber as hc
import inspect
import itertools


class ValidationException(Exception):
    """
    GAN components validate their configurations before creation.  
    
    `ValidationException` occcurs if they fail.
    """
    pass

class GANComponent:
    """
    GANComponents are pluggable pieces within a GAN.

    GAN objects are also GANComponents.
    """
    def __init__(self, gan, config):
        """
        Initializes a gan component based on a `gan` and a `config` dictionary.

        Different components require different config variables.  

        A `ValidationException` is raised if the GAN component configuration fails to validate.
        """
        self.gan = gan
        self.config = hc.Config(config)
        errors = self.validate()
        if errors != []:
            raise ValidationException(self.__class__.__name__+": " +"\n".join(errors))
        self.create_ops(config)

    def create_ops(self, config):
        """
        Create the ops object as `self.ops`.  Also looks up config
        """
        if self.gan is None:
            return
        if self.gan.ops_backend is None:
            return
        self.ops = self.gan.ops_backend(config=self.config, device=self.gan.device)
        self.config = self.gan.ops.lookup(config)

    def required(self):
        """
        Return a list of required config strings and a `ValidationException` will be thrown if any are missing.

        Example: 
        ```python
            class MyComponent(GANComponent):
                def required(self):
                    "learn rate is required"
                    ["learn_rate"]
        ```
        """
        return []

    def validate(self):
        """
        Validates a GANComponent.  Return an array of error messages. Empty array `[]` means success.
        """
        errors = []
        required = self.required()
        for argument in required:
            if(self.config.__getattr__(argument) == None):
                errors.append("`"+argument+"` required")

        if(self.gan is None):
            errors.append("GANComponent constructed without GAN")
        return errors

    def weights(self):
        """
            The weights of the GAN component.
        """
        return self.ops.weights

    def biases(self):
        """
            Biases of the GAN component.
        """
        return self.ops.biases

    def variables(self):
        """
            All variables associated with this component.
        """
        return self.ops.variables()

    def split_batch(self, net, count=2):
        """ 
        Discriminators return stacked results (on axis 0).  
        
        This splits the results.  Returns [d_real, d_fake]
        """
        ops = self.ops or self.gan.ops
        s = ops.shape(net)
        bs = s[0]
        nets = []
        net = ops.reshape(net, [bs, -1])
        start = [0 for x in ops.shape(net)]
        for i in range(count):
            size = [bs//count] + [x for x in ops.shape(net)[1:]]
            nets.append(ops.slice(net, start, size))
            start[0] += bs//count
        return nets

    def reuse(self, net):
        self.ops.reuse()
        net = self.build(net)
        self.ops.stop_reuse()
        return net

    def layer_regularizer(self, net):
        symbol = self.config.layer_regularizer
        op = self.gan.ops.lookup(symbol)
        if op:
            net = op(self, net)
        return net

    def split_by_width_height(self, net):
        elems = []
        ops = self.gan.ops
        shape = ops.shape(net)
        bs = shape[0]
        height = shape[1]
        width = shape[2]
        for i in range(width):
            for j in range(height):
                elems.append(ops.slice(net, [0, i, j, 0], [bs, 1, 1, -1]))

        return elems

    def permute(self, nets, k):
        return list(itertools.permutations(nets, k))

    #this is broken
    def fully_connected_from_list(self, nets):
        results = []
        ops = self.ops
        for net, net2 in nets:
            net = ops.concat([net, net2], axis=3)
            shape = ops.shape(net)
            bs = shape[0]
            net = ops.reshape(net, [bs, -1])
            features = ops.shape(net)[1]
            net = ops.linear(net, features)
            #net = self.layer_regularizer(net)
            net = ops.lookup('lrelu')(net)
            #net = ops.linear(net, features)
            net = ops.reshape(net, shape)
            results.append(net)
        return results

    def relation_layer(self, net):
        ops = self.ops

        #hack
        shape = ops.shape(net)
        input_size = shape[1]*shape[2]*shape[3]

        netlist = self.split_by_width_height(net)
        permutations = self.permute(netlist, 2)
        permutations = self.fully_connected_from_list(permutations)
        net = ops.concat(permutations, axis=3)

        #hack
        bs = ops.shape(net)[0]
        net = ops.reshape(net, [bs, -1])
        net = ops.linear(net, input_size)
        net = ops.reshape(net, shape)

        return net

