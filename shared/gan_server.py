from flask import Flask, send_file, request
import numpy as np
from shared.util import *
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

    def sample(self, type='batch', z=None, c=None, off=None):
        print("Creating sample")
        generator = get_tensor("g")
        y = get_tensor("y")
        x = get_tensor("x")
        z_t = get_tensor('z')
        f_t = get_tensor('f')
        eps_t = get_tensor('eps')
        print_z_t = get_tensor("print_z")
        categories = get_tensor('categories')


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
            categories_feed = np.reshape(categories_feed, categories[0].get_shape())

        rand = np.random.randint(0,self.config['y_dims'], size=self.config['batch_size'])
        rand = np.zeros_like(rand)
        print("Creating sample 2")
        random_one_hot = np.eye(self.config['y_dims'])[rand]

        sample_file = "samples/sample.png"
        if(type == 'batch'):
            sample = self.sess.run(generator, feed_dict={y:random_one_hot})
            print("Creating sample 3")
            stacks = [np.hstack(sample[x*8:x*8+8]) for x in range(8)]
            plot(self.config, np.vstack(stacks), sample_file)

        elif(type == 'feature'):

            z_t = get_tensor('z')
            z = self.sess.run(z_t)
            size = np.shape(z)[1]

            end = np.copy(z)[0]
            start = np.copy(z)[0]

            for i in c:
                i = int(i)
                end[i] = 2.0
                start[i] = -2.0
            print("Start", start, "End", end)

            zs = linspace(start, end)
            #print("zs",zs)
            print("Creating sample 3")
            sample = self.sess.run(generator, feed_dict={z_t:zs})
            print("Creating sample 3")
            stacks = [np.hstack(sample[x*8:x*8+8]) for x in range(8)]
            plot(self.config, np.vstack(stacks), sample_file)

        elif(type == 'linear'):
            encoded_z_t = get_tensor("encoded_z")

            def pick_best_f():
                f_t = get_tensor("f")
                d_fake_sigmoid_t = get_tensor("d_fake_sigmoid")
                eps_t = get_tensor('eps')
                z_t = get_tensor('z')
                fs = []
                for i in range(1):

                    [eps, d_fake_sigmoid, f, z] = self.sess.run([eps_t, d_fake_sigmoid_t, f_t, z_t], feed_dict={y:random_one_hot})

                    for f, d, e, z in zip(f, d_fake_sigmoid, eps, z):
                        fs.append({'f':f,'d':d,'e':e, 'z':z})
                    fs = sorted(fs, key=lambda x: (1-x['d']))
                print(" d sigmoid ", fs[0]['d'])
                return [fs[0]['f'], fs[0]['e'], fs[0]['z']]


            [start_f, start_eps, start_z] = pick_best_f()
            [end_f, end_eps, end_z] = pick_best_f()

            eps = linspace(start_eps, end_eps)
            f = linspace(start_f, end_f)
            z = linspace(start_z, end_z)
            #f = np.tile(start_f, [64, 1])
            #eps = np.zeros(eps_t.get_shape())
            #eps = np.random.normal(0,0.001, eps_t.get_shape())


            _,sample = self.sess.run([print_z_t, generator], feed_dict={f_t:f, eps_t: eps})
            stacks = [np.hstack(sample[x*8:x*8+8]) for x in range(8)]
            plot(self.config, np.vstack(stacks), sample_file)

        print("Sample ended", sample_file)
        return send_file(sample_file, mimetype='image/png')



def gan_server(sess, config):
    gws = GANWebServer(sess, config)
    @app.route('/sample.png')
    def sample():
        c =request.args.get('c')
        type = request.args.get('type')
        print('c is', c)
        if(c):
            c = c.split(',')
        print('c is now', c)
        return gws.sample(c=c, type=type)
    handler = RotatingFileHandler('server.log', maxBytes=10000, backupCount=1)
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    app.run()
