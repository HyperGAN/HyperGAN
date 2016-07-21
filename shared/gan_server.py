from flask import Flask, send_file, request
import numpy as np
from shared.util import *
import logging
from logging.handlers import RotatingFileHandler

app = Flask('gan')

class GANWebServer:
    def __init__(self, sess, config):
        self.sess = sess
        self.config = config

    def sample(self, type='batch', z=None, c=None):
        print("Creating sample")
        generator = get_tensor("g")
        y = get_tensor("y")
        x = get_tensor("x")
        z_t = get_tensor('z')
        f_t = get_tensor('f')
        print_z_t = get_tensor("print_z")
        categories = get_tensor('categories')


        categories_feed = []
        for i, category in enumerate(self.config['categories']):

            if(c and len(c) > i and c[i]):
                uc =int(c[i])
            else:
                uc = np.random.randint(0,category)
                #uc = 0
            categories_feed.append(np.eye(category)[uc])

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
        elif(type == 'linear'):
            encoded_z_t = get_tensor("encoded_z")
            start_z = self.sess.run(f_t, feed_dict={y:random_one_hot})
            start_z = start_z[0]
            end_z = self.sess.run(f_t, feed_dict={y:random_one_hot})
            end_z = end_z[0]

            #start_z = np.random.uniform(-1, 1, start_z.shape)
            #end_z = np.random.uniform(-1, 1, end_z.shape)
            print('start_z', np.shape(start_z))
            c = np.linspace(0,1, 64)
            a= np.array(start_z).reshape(-1,1)
            b= np.array(end_z).reshape(-1,1)
            z1 = a+ (b-a) * c
            z1 = np.transpose(z1)

            #z1 = [np.linspace(i, j, num=64) for i,j in zip(start_z, end_z)]
            #z1 = np.swapaxes(z1, 0,1)
            #z1 = np.transpose(np.vstack(z1))
            #z1 = np.transpose(np.vstack(z1))
            #z1 = np.random.uniform(-1,1,z_t.get_shape())
            #z1 = np.zeros(z_t.get_shape())
            
            _,sample = self.sess.run([print_z_t, generator], feed_dict={f_t:z1, y: random_one_hot})
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
