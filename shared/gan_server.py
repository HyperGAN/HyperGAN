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

    def sample(self, z=None, c=None):
        print("Creating sample")
        generator = get_tensor("g")
        y = get_tensor("y")
        x = get_tensor("x")
        z_t = get_tensor('z')
        categories = get_tensor('categories')


        categories_feed = []
        for i, category in enumerate(self.config['categories']):

            if(c and len(c) > i and c[i]):
                uc =int(c[i])
            else:
                uc = np.random.randint(0,category)
            categories_feed.append(np.eye(category)[uc])

        print("categories", categories_feed)
        categories_feed = np.hstack(categories_feed)
        categories_feed = np.tile(categories_feed, [self.config['batch_size'],1])
        categories_feed = np.reshape(categories_feed, categories.get_shape())

        rand = np.random.randint(0,self.config['y_dims'], size=self.config['batch_size'])
        rand = np.zeros_like(rand)
        print("Creating sample 2")
        random_one_hot = np.eye(self.config['y_dims'])[rand]
        sample = self.sess.run(generator, feed_dict={y:random_one_hot, categories: categories_feed})
        print("Creating sample 3")
        sample_file = "samples/sample.png"
        plot(self.config, sample[0], sample_file)
        print("Sample ended", sample_file)
        return send_file(sample_file, mimetype='image/png')



def gan_server(sess, config):
    gws = GANWebServer(sess, config)
    @app.route('/sample.png')
    def sample():
        c =request.args.get('c')
        print('c is', c)
        if(c):
            c = c.split(',')
        print('c is now', c)
        return gws.sample(c=c)
    handler = RotatingFileHandler('server.log', maxBytes=10000, backupCount=1)
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    app.run()
