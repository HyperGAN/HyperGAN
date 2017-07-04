import tensorflow as tf

def l2_distance(a,b):
    return tf.square(a-b)

def l1_distance(a,b):
    return tf.abs(a-b)
