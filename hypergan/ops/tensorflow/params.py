import tensorflow as tf

def decay(gan, *args):
    options = {
        "range": "0:1",
        "type": "linear",
        "steps": 10000,
        "on": 0
    }
    for arg in args:
        name, value = arg.split("=")
        options[name]=value
    r1,r2 = options["range"].split(":")
    r1 = float(r1)
    r2 = float(r2)
    cycle = "cycle" in options
    steps = int(options["steps"])
    onstep = int(options["on"])
    current_step = tf.train.get_global_step()
    type_ = options["type"]
    if type_ == "onoff-randomly":
        n = tf.random_uniform([1], minval=-1, maxval=1)
        if "offset" in options:
            n += tf.constant(float(options["offset"]), dtype=tf.float32)
        return (tf.sign(n) + 1) /2 * tf.constant(float(options["value"]), dtype=tf.float32)
    if onstep == 0:
        return tf.train.polynomial_decay(r1, current_step, steps, end_learning_rate=r2, power=1, cycle=cycle)
    else:
        onoff = tf.minimum(1.0, tf.cast(tf.nn.relu(current_step - onstep), tf.float32))
        return onoff * tf.train.polynomial_decay(r1, (current_step+onstep), steps, end_learning_rate=r2, power=1.0, cycle=cycle)
