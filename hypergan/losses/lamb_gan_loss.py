import tensorflow as tf
from hypergan.util.ops import *
from hypergan.util.hc_tf import *
import hyperchamber as hc

from hypergan.losses import wgan_loss, standard_gan_loss, lsgan_loss

def config(
        reduce=wgan_loss.linear_projection, 
        reverse=False,
        discriminator=None,
        label_smooth=list(np.linspace(0.15, 0.35, num=10)),
        alpha=0.001,
        beta=0.2,
        labels=[[0.5,0,-0.5]]
    ):
    selector = hc.Selector()
    selector.set("reduce", reduce)
    selector.set('reverse', reverse)
    selector.set('discriminator', discriminator)
    selector.set("label_smooth", label_smooth)
    selector.set('create', create)
    selector.set('alpha', alpha)
    selector.set('beta', beta)
    selector.set('labels', labels)

    return selector.random_config()

def create(config, gan):
    alpha = config.alpha
    beta = config.beta
    wgan_loss_d, wgan_loss_g = wgan_loss.create(config, gan)
    lsgan_loss_d, lsgan_loss_g = lsgan_loss.create(config, gan)
    config['reduce']=standard_gan_loss.linear_projection
    standard_loss_d, standard_loss_g = standard_gan_loss.create(config, gan)

    total = min(alpha + beta,1)

    d_loss = wgan_loss_d*alpha + lsgan_loss_d*beta + (1-total)*standard_loss_d
    g_loss = wgan_loss_g*alpha + lsgan_loss_g*beta + (1-total)*standard_loss_g

    return [d_loss, g_loss]

