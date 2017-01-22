#def loader():
#    # input stuff
#    graph = {
#            bw_x: bw_x,
#            x: x,
#            y: y
#            }
#    return graph


#TODO
def parse_args():

#TODO write
def add_bw(gan, net):
    return tf.greater()...

#TODO doesn't exist
initial_graph = hg.directory_loader() #TODO refactor from cli.py
#TODO parse args
args = parse_args()
config = hg.config.random(args)
config['generator.layer_filters'] = add_bw
# TODO same thing on D
gan = GAN(config, initial_graph)

gan.load_or_initialize_graph(save_file)

tf.train.start_queue_runners(sess=self.sess)
for i in range(100000):
    #TODO fix
    gan.train()

    #TODO sample b/w-> color
    gan.sample_to_file("samples/"+str(i)+".png")

tf.reset_default_graph()
self.sess.close()
