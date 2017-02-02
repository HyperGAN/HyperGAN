import hypergan
from .gan import GAN
from .graph import Graph
from .config import selector
import tensorflow as tf
import hypergan.cli

def load(name):
    gan = GAN()
    gan.load_config(name)
    #TODO it'd be nice to not do this
    args = cli.parse_args()
    channels = int(args.size.split("x")[2])
    width = int(args.size.split("x")[0])
    height = int(args.size.split("x")[1])
    gan.init_session(args.device)
    #TODO input system management?
    x,y,f,num_labels,examples_per_epoch = gan.setup_loader(
            args.format,
            args.directory,
            args.device,
            seconds=None,
            bitrate=None,
            width=width,
            height=height,
            channels=channels,
            crop=args.crop
    )
    #gan.create_graph('full')
    #TODO Are these saved?  confused here
    gan.config['y_dims']=num_labels
    gan.config['x_dims']=[height,width] #todo can we remove this?
    gan.config['channels']=channels
    gan.config['batch_size']=args.batch_size
    gan.config['dtype']=tf.float32
    graph = gan.create_graph(x, y, f, 'full', args.device)
    #TODO loading

    return gan

def load_generator(name):
    gan = GAN()
    gan.load_config(name)
    gan.create_graph('generator')
    return gan
