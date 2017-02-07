def config():
    selector = hc.Selector()
    selector.set("reduce", [tf.reduce_mean])#reduce_sum, reduce_logexp work

    selector.set('create', create)
    
    return selector.random_config()

def create(config, gan):
    d_real_lin = config.reduce(d_real_lin, axis=1)
    d_fake_lin = config.reduce(d_fake_lin, axis=1)
    d_loss = d_real_lin - d_fake_lin
    g_loss = d_fake_lin
    d_fake_loss = -d_fake_lin
    d_real_loss = d_real_lin

    return [d_loss, g_loss]




