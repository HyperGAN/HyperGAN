import hyperchamber as hc
import tensorflow as tf
import importlib

from hypergan.discriminators import *
from hypergan.encoders import *
from hypergan.generators import *
from hypergan.regularizers import *
from hypergan.samplers import *
from hypergan.trainers import *
from hypergan.util import *

# Below are sets of configuration options:
# Each time a new random network is started a random set of configuration variables are selected.
# This is useful for hyperparameter search.  If you want to use a specific configuration use --config

def selector(args):
    selector = hc.Selector()
    selector.set('dtype', tf.float32) #The data type to use in our GAN.  Only float32 is supported at the moment

    # Z encoder configuration
    selector.set('encoder', random_combo_encoder.encode_periodic_gaussian) # how to encode z

    # Generator configuration
    selector.set("generator.z", 40) # the size of the encoding.  Encoder is set by the 'encoder' property, but could just be a random_uniform
    selector.set("generator", [resize_conv.generator])
    selector.set("generator.z_projection_depth", 512) # Used in the first layer - the linear projection of z
    selector.set("generator.activation", [prelu("g_")]); # activation function used inside the generator
    selector.set("generator.activation.end", [tf.nn.tanh]); # Last layer of G.  Should match the range of your input - typically -1 to 1
    selector.set("generator.fully_connected_layers", 0) # Experimental - This should probably stay 0
    selector.set("generator.final_activation", [tf.nn.tanh]) #This should match the range of your input
    selector.set("generator.resize_conv.depth_reduction", 2) # Divides our depth by this amount every time we go up in size
    selector.set('generator.layer.noise', False) #Adds incremental noise each layer
    selector.set('generator.layer_filter', None) #Add information to g
    selector.set("generator.regularizers.l2.lambda", list(np.linspace(0.1, 1, num=30))) # the magnitude of the l2 regularizer(experimental)
    selector.set("generator.regularizers.layer", [batch_norm_1]) # the magnitude of the l2 regularizer(experimental)
    selector.set('generator.densenet.size', 16)
    selector.set('generator.densenet.layers', 1)

    # Trainer configuration
    #trainer = wgan_trainer # adam works well at 64x64 but doesn't scale
    trainer = adam_trainer # adam works well at 64x64 but doesn't scale
    #trainer = slowdown_trainer # this works at higher resolutions, but is slow and quirky(help wanted)
    #trainer = rmsprop_trainer # this works well with wgan
    #trainer = sgd_adam_trainer # This has never worked, but seems like it should
    selector.set("trainer.initializer", trainer.initialize) # TODO: can we merge these variables?
    selector.set("trainer.train", trainer.train) # The training method to use.  This is called every step
    selector.set("trainer.rmsprop.discriminator.lr", 1e-4) # d learning rate
    selector.set("trainer.rmsprop.generator.lr", 1e-4) # g learning rate
    selector.set("trainer.adam.discriminator.lr", 1e-3) #adam_trainer d learning rate
    selector.set("trainer.adam.discriminator.epsilon", 1e-8) #adam epsilon for d
    selector.set("trainer.adam.discriminator.beta1", 0.9) #adam beta1 for d
    selector.set("trainer.adam.discriminator.beta2", 0.999) #adam beta2 for d
    selector.set("trainer.adam.generator.lr", 1e-3) #adam_trainer g learning rate
    selector.set("trainer.adam.generator.epsilon", 1e-8) #adam_trainer g
    selector.set("trainer.adam.generator.beta1", 0.9) #adam_trainer g
    selector.set("trainer.adam.generator.beta2", 0.999) #adam_trainer g
    selector.set('trainer.slowdown.discriminator.d_fake_min', [0.12]) # healthy above this number on d_fake
    selector.set('trainer.slowdown.discriminator.d_fake_max', [0.12001]) # unhealthy below this number on d_fake
    selector.set('trainer.slowdown.discriminator.slowdown', [5]) # Divides speed by this number when unhealthy(d_fake low)
    selector.set("trainer.sgd_adam.discriminator.lr", 3e-4) # d learning rate
    selector.set("trainer.sgd_adam.generator.lr", 1e-3) # g learning rate

    # TODO: cleanup
    selector.set("examples_per_epoch", 30000/4)

    # Discriminator configuration
    discriminators = []
    for i in range(1):
        discriminators.append(pyramid_nostride_fc_discriminator.config(layers=5))
    selector.set("discriminators", [discriminators])

    # Sampler configuration
    selector.set("sampler", progressive_enhancement_sampler.sample) # this is our sampling method.  Some other sampling ideas include cosine distance or adverarial encoding(not implemented but contributions welcome).
    selector.set("sampler.samples", 3) # number of samples to generate at the end of each epoch

    selector.set('categories', [[]])
    selector.set('categories_lambda', list(np.linspace(.001, .01, num=100)))
    selector.set('category_loss', [False])

    # Loss function configuration
    selector.set('g_class_loss', [False])
    selector.set('g_class_lambda', list(np.linspace(0.01, .1, num=30)))

    selector.set("g_target_prob", list(np.linspace(.65 /2., .85 /2., num=100)))
    selector.set("d_label_smooth", list(np.linspace(0.15, 0.35, num=100)))

    return selector

def random(args):
    return selector(args).random_config()


# This looks up a function by name.   Should it be part of hyperchamber?
#TODO moveme
def get_function(name):
    if name == "function:hypergan.util.ops.prelu_internal":
        return prelu("g_")

    if not isinstance(name, str):
        return name
    namespaced_method = name.split(":")[1]
    method = namespaced_method.split(".")[-1]
    namespace = ".".join(namespaced_method.split(".")[0:-1])
    return getattr(importlib.import_module(namespace),method)

# Take a config and replace any string starting with 'function:' with a function lookup.
#TODO moveme
def lookup_functions(config):
    for key, value in config.items():
        if(isinstance(value, str) and value.startswith("function:")):
            config[key]=get_function(value)
        if(isinstance(value, list) and len(value) > 0 and isinstance(value[0],str) and value[0].startswith("function:")):
            config[key]=[get_function(v) for v in value]

    return config


