import tensorflow as tf

def capped_optimizer(optimizer, lr, loss, vars):
  capped_optimizer = optimizer(lr)
  gvs = capped_optimizer.compute_gradients(loss, var_list=vars)
  def create_cap(grad,var):
    if(grad == None):
        print("Warning: No gradient for variable ",var.name)
        return None
    return (tf.clip_by_value(grad, -1, 1), var)
  capped_gvs = [create_cap(grad,var) for grad, var in gvs]
  capped_gvs = [x for x in capped_gvs if x != None]
  return capped_optimizer.apply_gradients(capped_gvs)
