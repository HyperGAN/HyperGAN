import tensorflow as tf
import numpy as np
import hyperchamber as hc

from hypergan.losses.base_loss import BaseLoss

TINY=1e-8
class FDivergenceLoss(BaseLoss):

    def _create(self, d_real, d_fake):
        gan = self.gan
        config = self.config

        gfx = None
        gfg = None

        pi = config.pi or 2

        g_loss_type = config.g_loss_type or 'gan'

        alpha = config.alpha or 0.5

        if config.type == 'kl':
            bounded_x = tf.minimum(tf.constant(np.exp(9), dtype=tf.float32), d_real)
            bounded_g = tf.minimum(10., d_fake)
            gfx = bounded_x
            gfg = bounded_g
        elif config.type == 'js':
            gfx = np.log(2) - tf.log(1+tf.exp(-d_real))
            gfg = np.log(2) - tf.log(1+tf.exp(-d_fake))
        elif config.type == 'js_weighted':
            gfx = -pi*np.log(pi) - tf.log(1+tf.exp(-d_real))
            gfg = -pi*np.log(pi) - tf.log(1+tf.exp(-d_fake))
        elif config.type == 'gan':
            gfx = -tf.log(1+tf.exp(-d_real))
            gfg = -tf.log(1+tf.exp(-d_fake))
        elif config.type == 'reverse_kl':
            gfx = -tf.exp(-d_real)
            gfg = -tf.exp(-d_fake)
        elif config.type == 'pearson' or config.type == 'jeffrey' or config.type == 'alpha2':
            gfx = d_real
            gfg = d_fake
        elif config.type == 'squared_hellinger':
            gfx = 1-tf.exp(-d_real)
            gfg = 1-tf.exp(-d_fake)
        elif config.type == 'neyman':
            gfx = 1-tf.exp(-d_real)
            gfx = tf.minimum(gfx, 1.9)
            gfg = 1-tf.exp(-d_fake)

        elif config.type == 'total_variation':
            gfx = 0.5*tf.nn.tanh(d_real)
            gfg = 0.5*tf.nn.tanh(d_fake)

        elif config.type == 'alpha1':
            gfx = 1./(1-alpha) - tf.log(1+tf.exp(-d_real))
            gfg = 1./(1-alpha) - tf.log(1+tf.exp(-d_fake))

        else:
            raise "Unknown type " + config.type

        conjugate = None

        if config.type == 'kl':
            conjugate = tf.exp(gfg-1)
        elif config.type == 'js':
            bounded = tf.minimum(tf.log(2.)-TINY, gfg)
            conjugate = -tf.log(2-tf.exp(bounded))
        elif config.type == 'js_weighted':
            c = -pi*np.log(pi)-TINY
            c = tf.constant(c, dtype=tf.float32)
            bounded = gfg#tf.maximum(gfg, c)
            conjugate = (1-pi)*tf.log((1-pi)/((1-pi)*tf.exp(bounded/pi)))
        elif config.type == 'gan':
            conjugate = -tf.log(1-tf.exp(gfg))
        elif config.type == 'reverse_kl':
            conjugate = -1-tf.log(-gfg)
        elif config.type == 'pearson':
            conjugate = 0.25 * tf.square(gfg)+gfg
        elif config.type == 'neyman':
            conjugate = 2 - 2 * tf.sqrt(tf.nn.relu(1-gfg)+1e-2)
        elif config.type == 'squared_hellinger':
            conjugate = gfg/(1.-gfg)
        elif config.type == 'jeffrey':
            raise "jeffrey conjugate not implemented"

        elif config.type == 'alpha2' or config.type == 'alpha1':
            bounded = gfg
            bounded = 1./alpha * (bounded * ( alpha - 1) + 1)
            conjugate = tf.pow(bounded, alpha/(alpha - 1.)) - 1. / alpha

        elif config.type == 'total_variation':
            conjugate = gfg
        else:
            raise "Unknown type " + config.type

        gf_threshold  = None # f' in the paper

        if config.type == 'kl':
            gf_threshold = 1
        elif config.type == 'js':
            gf_threshold = 0
        elif config.type == 'gan':
            gf_threshold = -np.log(2)
        elif config.type == 'reverse_kl':
            gf_threshold = -1
        elif config.type == 'pearson':
            gf_threshold = 0
        elif config.type == 'squared_hellinger':
            gf_threshold = 0

        self.gf_threshold=gf_threshold

        d_loss = -gfx+conjugate
        g_loss = -conjugate

        if g_loss_type == 'gan':
            g_loss = -conjugate
        elif g_loss_type == 'total_variation':
            # The inverse of derivative(1/2*x - 1)) = 0.5
            # so we use the -conjugate for now
            g_loss = -conjugate
        elif g_loss_type == 'js':
            # https://www.wolframalpha.com/input/?i=inverse+of+derivative(-(u%2B1)*log((1%2Bu)%2F2)%2Bu*log(u))
            g_loss = -tf.exp(d_fake)
        elif g_loss_type == 'js_weighted':
            # https://www.wolframalpha.com/input/?i=inverse+of+derivative(-(u%2B1)*log((1%2Bu)%2F2)%2Bu*log(u))
            p = pi
            u = d_fake
            #inner = (-4.*u*tf.exp(p/u) + tf.exp(2.)*tf.square(u)-2.*tf.exp(2.)*u+tf.exp(2.))/tf.square(u)
            #inner = tf.nn.relu(inner) + 1e-3
            #g_loss = (1.-u)/(2.*u) - tf.sqrt(inner)/(2.*tf.exp(1.))
            exp_bounded = p/u
            exp_bounded = tf.minimum(4., exp_bounded)
            inner = (-4.*u*tf.exp(exp_bounded) +np.exp(2.)*tf.square(u)-2.*np.exp(2.)*u+np.exp(2.))/tf.square(u)
            inner = tf.nn.relu(inner)
            u = tf.maximum(0.1,u)
            sqrt = tf.sqrt(inner+1e-2) / (2*np.exp(1))
            g_loss = (1.-u)/(2.*u)# + sqrt
        elif g_loss_type == 'pearson':
            g_loss = -(d_fake-2.0)/2.0
        elif g_loss_type == 'neyman':
            g_loss = 1./tf.sqrt(1-d_fake) # does not work, causes 'nan'
        elif g_loss_type == 'squared_hellinger':
            g_loss = -1.0/(tf.square(d_fake-1)+1e-2)
        elif g_loss_type == 'reverse_kl':
            g_loss = -d_fake
        elif g_loss_type == 'kl':
            g_loss = -gfg * tf.exp(gfg)
        elif g_loss_type == 'alpha1': 
            a = alpha
            bounded = d_fake
            g_loss = (1.0/(a*(a-1))) * (tf.exp(a*bounded) - 1 - a*(tf.exp(bounded) - 1))
        elif g_loss_type == 'alpha2':
            a = alpha
            bounded = tf.minimum(d_fake, 4.)
            g_loss = -(1.0/(a*(a-1))) * (tf.exp(a*bounded) - 1 - a*(tf.exp(bounded) - 1))
        else:
            raise "Unknown g_loss_type " + config.type

        self.gfg = gfg
        self.gfx = gfx

        self.d_real = d_real
        self.d_fake = d_fake

        return [d_loss, g_loss]

    def g_regularizers(self):
        regularizer = None
        config = self.config
        pi = config.pi or 2
        alpha = config.alpha or 0.5

        ddfc = 0

        if config.regularizer is None:
            return []
        if config.regularizer == 'kl':
            # https://www.wolframalpha.com/input/?i=derivative(derivative(exp(t-1)))
            bounded = tf.minimum(4., self.gfg)
            ddfc = tf.exp(bounded - 1)
        elif config.regularizer == 'js':
            # https://www.wolframalpha.com/input/?i=derivative(derivative(-log(2-exp(t))))
            ddfc = -(2*tf.exp(self.gfg)) / (tf.square(2-tf.exp(self.gfg))+1e-2)
        elif config.regularizer == 'js_weighted':
            # https://www.wolframalpha.com/input/?i=derivative(derivative((1-C)*log(((1-C)%2F(1-C*exp(x%2FC)))))
            ddfc = -((pi-1)*tf.exp(self.gfg/pi))/(pi*tf.square(pi*tf.exp(self.gfg/pi)-1))
        elif config.regularizer == 'gan':
            # https://www.wolframalpha.com/input/?i=derivative(derivative(-log(1-exp(t))))
            ddfc = (2*tf.exp(self.gfg)) / (tf.square(1-tf.exp(self.gfg))+1e-2)
        elif config.regularizer == 'reverse_kl':
            # https://www.wolframalpha.com/input/?i=derivative(derivative(-1-log(-x)))
            ddfc = 1.0/tf.square(self.gfg)
        elif config.regularizer == 'pearson':
            # https://www.wolframalpha.com/input/?i=derivative(derivative(0.25*x*x+%2B+x))
            ddfc = 0.5
        elif config.regularizer == 'jeffrey':
            raise "jeffrey regularizer not implemented"
        elif config.regularizer == 'squared_hellinger': 
            # https://www.wolframalpha.com/input/?i=derivative(derivative(t%2F(1-t)))
            ddfc = 2 / (tf.pow(self.gfg - 1, 3)+1e-2)
            #ddfc = 0
        elif config.regularizer == 'neyman':
            # https://www.wolframalpha.com/input/?i=derivative(derivative(2-2*sqrt(1-t)))
            ddfc = 1.0/(2*tf.pow(1-self.gfg, 3/2))
        elif config.regularizer == 'total_variation':
            # https://www.wolframalpha.com/input/?i=derivative(derivative(t))
            ddfc = 0
        elif config.regularizer == 'alpha1' or config.regularizer == 'alpha2':
            # https://www.wolframalpha.com/input/?i=derivative(derivative(1%2FC*(x*(C-1)%2B1)%5E(C%2FC-1)-1%2FC))
            ddfc = -tf.pow((alpha - 1)*self.gfg+1, (1/(alpha-1)-1))
        regularizer = ddfc * tf.nn.l2_normalize(self.gfg, [0]) * (config.regularizer_lambda or 1)
        self.metrics['fgan_regularizer'] = self.gan.ops.squash(regularizer)
        return [regularizer ]
        
    def d_regularizers(self):
        return []
