import tensorflow as tf
import numpy as np

def l2_distance(a,b):
    return tf.square(a-b)

def l1_distance(a,b):
    return tf.abs(a-b)

def bicubic_interp_2d(input_, new_size, endpoint=False):
  """
  Args :
    input_ : Input tensor. Its shape should be
        [batch_size, height, width, channel].
        In this implementation, the shape should be fixed for speed.
    new_size : The output size [new_height, new_width]
  ref : 
    http://blog.demofox.org/2015/08/15/resizing-images-with-bicubic-interpolation/
  """

  shape = input_.get_shape().as_list()
  batch_size = shape[0]
  height  = shape[1]
  width   = shape[2]
  channel = shape[3]
 
  def _hermite(A, B, C, D, t):
    a = A * (-0.5) + B * 1.5 + C * (-1.5) + D * 0.5
    b = A + B * (-2.5) + C * 2.0 + D * (-0.5)
    c = A * (-0.5) + C * 0.5
    d = B

    return a*t*t*t + b*t*t + c*t + d

  def _get_grid_array(n_i, y_i, x_i, c_i):
    n, y, x, c = np.meshgrid(n_i, y_i, x_i, c_i, indexing='ij')
    n = np.expand_dims(n, axis=4)
    y = np.expand_dims(y, axis=4)
    x = np.expand_dims(x, axis=4)
    c = np.expand_dims(c, axis=4)
    
    return np.concatenate([n,y,x,c], axis=4)

  def _get_frac_array(y_d, x_d, n, c):
    y = y_d.shape[0]
    x = x_d.shape[0]
    y_t = y_d.reshape([1, -1, 1, 1])
    x_t = x_d.reshape([1, 1, -1, 1])
    y_t = tf.constant(np.tile(y_t, (n,1,x,c)), dtype=tf.float32)
    x_t = tf.constant(np.tile(x_t, (n,y,1,c)), dtype=tf.float32)
    return y_t, x_t

  def _get_index_tensor(grid, x, y):
    new_grid = np.array(grid)

    grid_y = grid[:,:,:,:,1] + y
    grid_x = grid[:,:,:,:,2] + x

    grid_y = np.clip(grid_y, 0, height-1)
    grid_x = np.clip(grid_x, 0, width-1)

    new_grid[:,:,:,:,1] = grid_y
    new_grid[:,:,:,:,2] = grid_x

    return tf.constant(new_grid, dtype=tf.int32)

  new_height = new_size[0]
  new_width  = new_size[1]

  n_i = np.arange(batch_size)
  c_i = np.arange(channel)

  if endpoint:
    y_f = np.linspace(0., height-1, new_height)
  else:
    y_f = np.linspace(0., height, new_height, endpoint=False)
  y_i = y_f.astype(np.int32)
  y_d = y_f - np.floor(y_f)

  if endpoint:
    x_f = np.linspace(0., width-1, new_width)
  else:
    x_f = np.linspace(0., width, new_width, endpoint=False)
  x_i = x_f.astype(np.int32)
  x_d = x_f - np.floor(x_f) 

  grid = _get_grid_array(n_i, y_i, x_i, c_i)
  y_t, x_t = _get_frac_array(y_d, x_d, batch_size, channel)

  i_00 = _get_index_tensor(grid, -1, -1)
  i_10 = _get_index_tensor(grid, +0, -1)
  i_20 = _get_index_tensor(grid, +1, -1)
  i_30 = _get_index_tensor(grid, +2, -1)
      
  i_01 = _get_index_tensor(grid, -1, +0)
  i_11 = _get_index_tensor(grid, +0, +0)
  i_21 = _get_index_tensor(grid, +1, +0)
  i_31 = _get_index_tensor(grid, +2, +0)
      
  i_02 = _get_index_tensor(grid, -1, +1)
  i_12 = _get_index_tensor(grid, +0, +1)
  i_22 = _get_index_tensor(grid, +1, +1)
  i_32 = _get_index_tensor(grid, +2, +1)
      
  i_03 = _get_index_tensor(grid, -1, +2)
  i_13 = _get_index_tensor(grid, +0, +2)
  i_23 = _get_index_tensor(grid, +1, +2)
  i_33 = _get_index_tensor(grid, +2, +2)

  p_00 = tf.gather_nd(input_, i_00)
  p_10 = tf.gather_nd(input_, i_10)
  p_20 = tf.gather_nd(input_, i_20)
  p_30 = tf.gather_nd(input_, i_30)

  p_01 = tf.gather_nd(input_, i_01)
  p_11 = tf.gather_nd(input_, i_11)
  p_21 = tf.gather_nd(input_, i_21)
  p_31 = tf.gather_nd(input_, i_31)

  p_02 = tf.gather_nd(input_, i_02)
  p_12 = tf.gather_nd(input_, i_12)
  p_22 = tf.gather_nd(input_, i_22)
  p_32 = tf.gather_nd(input_, i_32)

  p_03 = tf.gather_nd(input_, i_03)
  p_13 = tf.gather_nd(input_, i_13)
  p_23 = tf.gather_nd(input_, i_23)
  p_33 = tf.gather_nd(input_, i_33)

  col0 = _hermite(p_00, p_10, p_20, p_30, x_t)
  col1 = _hermite(p_01, p_11, p_21, p_31, x_t)
  col2 = _hermite(p_02, p_12, p_22, p_32, x_t)
  col3 = _hermite(p_03, p_13, p_23, p_33, x_t)
  value = _hermite(col0, col1, col2, col3, y_t)
  
  return value
