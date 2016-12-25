from flask import Flask, send_file, request
import numpy as np
from hypergan.util import *
from hypergan.util.globals import *
import logging
from logging.handlers import RotatingFileHandler

app = Flask('gan')

def linspace(start, end):
    c = np.linspace(0,1, 64)
    a= np.array(start).reshape(-1,1)
    b= np.array(end).reshape(-1,1)
    f = a+ (b-a) * c
    f = np.transpose(f)
    return f

class GANWebServer:
    def __init__(self, sess, config):
        self.sess = sess
        self.config = config
        self.seed_bank = {}

    def random_one_hot(self):
        rand = np.random.randint(0,self.config['y_dims'], size=self.config['batch_size'])
        rand = np.zeros_like(rand)

        return np.eye(self.config['y_dims'])[rand]

    def sample_batch(self, sample_file):
        generator = get_tensor("g")[-1]
        y_t = get_tensor("y")
        print("generator is ", generator)
        sample = self.sess.run(generator, feed_dict={y_t:self.random_one_hot()})
        print("sample is ", sample)
        print(sample.shape)

        stacks = [np.hstack(sample[x*6:x*6+6]) for x in range(4)]
        plot(self.config, np.vstack(stacks), sample_file)
       
       #plot(self.config, sample, sample_file)


    def sample_grid(self, sample_file):
        generator = get_tensor("g")[-1]
        y_t = get_tensor("y")
        z_t = get_tensor("z")


        x = np.linspace(0,1, 4)
        y = np.linspace(0,1, 6)

        #z = np.mgrid[-3:3:0.75, -3:3:0.38*3].reshape(2,-1).T
        #z = np.mgrid[-3:3:0.6*3, -3:3:0.38*3].reshape(2,-1).T
        #z = np.mgrid[-6:6:0.6*6, -6:6:0.38*6].reshape(2,-1).T

        z = np.mgrid[-1:1:0.6, -1:1:0.38].reshape(2,-1).T
        #z = np.mgrid[0:1000:300, 0:1000:190].reshape(2,-1).T
        #z = np.mgrid[-0:1:0.3, 0:1:0.19].reshape(2,-1).T
        #z = np.mgrid[0.25:-0.25:-0.15, 0.25:-0.25:-0.095].reshape(2,-1).T
        #z = np.mgrid[-0.125:0.125:0.075, -0.125:0.125:0.095/2].reshape(2,-1).T
        #z = np.zeros(z_t.get_shape())
        #z.fill(0.2)

        print(z)

        sample = self.sess.run(generator, feed_dict={y_t:self.random_one_hot(), z_t: z})
        #plot(self.config, sample, sample_file)
        stacks = [np.hstack(sample[x*6:x*6+6]) for x in range(4)]
        plot(self.config, np.vstack(stacks), sample_file)

    def sample_iterate_z(self, sample_file, z_iterate, target_value=1, seed=None):
        generator = get_tensor("g")
        z_t = get_tensor('z')
        if(seed in self.seed_bank):
            print("Found z in bank")
            z = self.seed_bank[seed]
        else:
            print("New z")
            z = self.sess.run(z_t)
            self.seed_bank[seed] = z
            
        y_t = get_tensor("y")
        size = np.shape(z)[1]

        #z = np.random.uniform(-1,1, np.shape(z))
        z_iterate = [int(x) for x in z_iterate]
        def val(elem):
            vals = []
            for i,x in enumerate(elem):
                if(i in z_iterate):
                    vals.append(np.float32(target_value))
                else:
                    vals.append(0)
            return vals
        z = z+np.array([val(elem) for elem in z])

        sample = self.sess.run(generator, feed_dict={z_t:z,y_t:self.random_one_hot()})
        stacks = [np.hstack(sample[x*8:x*8+8]) for x in range(8)]
        plot(self.config, np.vstack(stacks), sample_file)

    def pick_best_f(self):
        f_t = get_tensor("f")
        d_fake_sigmoid_t = get_tensor("d_fake_sigmoid")
        eps_t = get_tensor('eps')
        z_t = get_tensor('z')
        y_t = get_tensor("y")
        fs = []
        for i in range(1):

            [eps, d_fake_sigmoid, f, z] = self.sess.run(
                                [eps_t, d_fake_sigmoid_t, f_t, z_t], 
                                feed_dict={y_t:self.random_one_hot()})

            for f, d, e, z in zip(f, d_fake_sigmoid, eps, z):
                fs.append({'f':f,'d':d,'e':e, 'z':z})
            fs = sorted(fs, key=lambda x: (1-x['d']))
        print(" d sigmoid ", fs[0]['d'])
        return [fs[0]['f'], fs[0]['e'], fs[0]['z']]


    def sample_feature(self, sample_file):
        encoded_z_t = get_tensor("encoded_z")
        print_z_t = get_tensor("print_z")
        generator = get_tensor("g")
        f_t = get_tensor("f")
        eps_t = get_tensor("eps")


        [start_f, start_eps, start_z] = self.pick_best_f()
        [end_f, end_eps, end_z] = self.pick_best_f()

        eps = linspace(start_eps, end_eps)
        f = linspace(start_f, end_f)
        z = linspace(start_z, end_z)
        #f = np.tile(start_f, [64, 1])
        #eps = np.zeros(eps_t.get_shape())
        #eps = np.random.normal(0,0.001, eps_t.get_shape())


        _,sample = self.sess.run([print_z_t, generator], feed_dict={f_t:f, eps_t: eps})
        stacks = [np.hstack(sample[x*8:x*8+8]) for x in range(8)]
        plot(self.config, np.vstack(stacks), sample_file)


    def sample(self, type='batch', c=None, features=None, z_iterate=None, target_value=None, seed=None,should_send_file=True):
        print("Creating sample")

        categories_feed = []
        for i, category in enumerate(self.config['categories']):
            if(c and len(c) > i and c[i]):
                uc =int(c[i])
            else:
                uc = np.random.randint(0,category)
                uc = 0
            categories_feed.append(np.eye(category)[uc])

        if len(categories_feed) > 0:
            categories_feed = np.hstack(categories_feed)
            categories_feed = np.tile(categories_feed, [self.config['batch_size'],1])


        sample_file = "sample.png"
        if(type == 'batch'):
            self.sample_batch(sample_file)
        elif(type == 'feature'):
            self.sample_iterate_z(sample_file, z_iterate, target_value, seed)
        elif(type == 'linear'):
            self.sample_feature(sample_file)
        elif(type == 'grid'):
            self.sample_grid(sample_file)
        print("Sample ended", sample_file)
        if(should_send_file):
            return send_file(sample_file, mimetype='image/png')
        else:
            return sample_file



def gan_sample(sess, config):
    gws = GANWebServer(sess, config)
    type='batch'
    c=[]
    z_iterate=[]
    target=0
    seed=0
    return gws.sample(c=c, type=type, z_iterate=z_iterate, target_value=target, seed=seed, send_file=False)

def gan_server(sess, config):
    gws = GANWebServer(sess, config)
    @app.route('/sample.png')
    def sample():
        c =request.args.get('c')
        type = request.args.get('type')
        z_iterate = request.args.get('z_iterate')
        target = request.args.get('target')
        seed = request.args.get('seed')
        print('c is', c)
        if(c):
            c = c.split(',')
    
        if(z_iterate):
            z_iterate = z_iterate.split(',')
        print('c is now', c)
        return gws.sample(c=c, type=type, z_iterate=z_iterate, target_value=target, seed=seed)
    handler = RotatingFileHandler('server.log', maxBytes=10000, backupCount=1)
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    app.run(host='0.0.0.0')
