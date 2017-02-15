from flask import Flask, send_file, request
from flask_cors import CORS, cross_origin
import numpy as np
from PIL import Image
from hypergan.util import *
from hypergan.samplers.common import *
from hypergan.samplers import grid_sampler
import logging
import json
import re
from logging.handlers import RotatingFileHandler

app = Flask('gan')

CORS(app)
import base64
from io import BytesIO, StringIO

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
        generator = gan.graph.g[-1]
        y_t = gan.graph.y
        print("generator is ", generator)

        #TODO classes broken 
        # y_t:random_one_hot(see git log)
        sample = self.sess.run(generator, feed_dict={})
        print("sample is ", sample)
        print(sample.shape)

        stacks = [np.hstack(sample[x*6:x*6+6]) for x in range(4)]
        plot(self.config, np.vstack(stacks), sample_file)
       
       #plot(self.config, sample, sample_file)

    def sample_zeros(self, sample_file):
        generator = gan.graph.g[-1]
        #categories_t = gan.graph.categories")[
        y_t = gan.graph.y
        z_t = gan.graph.z
        z = np.ones(z_t.get_shape())*2
        #categories = np.zeros(categories_t.get_shape())
        print("generator is ", generator)
        #TODO classes broken 
        # y_t:random_one_hot(see git log)

        sample = self.sess.run(generator, feed_dict={ z_t: z})

        stacks = [np.hstack(sample[x*8:x*8+8]) for x in range(4)]
        plot(self.config, np.vstack(stacks), sample_file)
     
    def sample_grid(self, sample_file):
        grid_sampler.sample(sample_file, self.sess, self.config)

    def sample_iterate_z(self, sample_file, z_iterate, target_value=1, seed=None):
        generator = gan.graph.g
        z_t = gan.graph.z
        if(seed in self.seed_bank):
            print("Found z in bank")
            z = self.seed_bank[seed]
        else:
            print("New z")
            z = self.sess.run(z_t)
            self.seed_bank[seed] = z
            
        y_t = gan.graph.y
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
        f_t = gan.graph.f
        d_fake_sigmoid_t = gan.graph.d_fake_sigmoid
        eps_t = gan.graph.eps
        z_t = gan.graph.z
        y_t = gan.graph.y
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
        encoded_z_t = gan.graph.encoded_z
        print_z_t = gan.graph.print_z
        generator = gan.graph.g
        f_t = gan.graph.f
        eps_t = gan.graph.eps


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

    def sample_base64(self, sample_file, x):
        generator = get_tensor("g")[-1]
        y_t = get_tensor("y")
        x_t = get_tensor("x")

        if x is not None:
            print("LEN X", len(x))
            x = base64.b64decode(bytes(re.sub('^data:image/.+;base64,', '', x), 'ascii'))
            f = open("x.png", "wb")
            f.write(x)
            f.close()
            x = Image.open('x.png')
            x = np.asarray(x, dtype='uint8')
            x = x/127.5-1
            x = x.reshape([1, x.shape[0], x.shape[1], x.shape[2]])
            x = np.tile(x, [self.config['batch_size'],1,1,1])
            sample = self.sess.run(generator, feed_dict={x_t:x})
        else:
            print("x is None")
            sample = self.sess.run(generator, feed_dict={})

        stacks = [np.hstack(sample[x*8:x*8+8]) for x in range(4)]
        plot(self.config, np.vstack(stacks), sample_file)

    def sample(self, type='batch', c=None, features=None, z_iterate=None, target_value=None, seed=None,should_send_file=True,x=None):
        print("Creating sample")

        #categories_feed = []
        #for i, category in enumerate(self.config['categories']):
        #    if(c and len(c) > i and c[i]):
        #        uc =int(c[i])
        #    else:
        #        uc = np.random.randint(0,category)
        #        uc = 0
        #    categories_feed.append(np.eye(category)[uc])

        #if len(categories_feed) > 0:
        #    categories_feed = np.hstack(categories_feed)
        #    categories_feed = np.tile(categories_feed, [self.config['batch_size'],1])


        sample_file = "sample.png"
        if(type == 'batch'):
            self.sample_batch(sample_file)
        elif(type == 'feature'):
            self.sample_iterate_z(sample_file, z_iterate, target_value, seed)
        elif(type == 'linear'):
            self.sample_feature(sample_file)
        elif(type == 'grid'):
            self.sample_grid(sample_file)
        elif(type == 'zero'):
            self.sample_zeros(sample_file)
        elif(type == 'base64'):
            self.sample_base64(sample_file, x)
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
    @app.route('/sample.json', methods=['POST', 'GET'])
    def sampleJson():
        x = request.json['x']
        gws.sample_base64('x.png', x)
        return send_file('x.png', mimetype='image/png')

    @app.route('/sample.png')
    def sample():
        c =request.args.get('c')
        type = request.args.get('type')
        z_iterate = request.args.get('z_iterate')
        target = request.args.get('target')
        seed = request.args.get('seed')
        x = request.args.get('x')
        print('c is', c)
        if(c):
            c = c.split(',')
    
        if(z_iterate):
            z_iterate = z_iterate.split(',')
        print('c is now', c)
        return gws.sample(c=c, type=type, z_iterate=z_iterate, target_value=target, seed=seed,x=x)
    handler = RotatingFileHandler('server.log', maxBytes=10000, backupCount=1)
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    app.run(host='0.0.0.0')
