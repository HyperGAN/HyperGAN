
# coding: utf-8

# In[1]:

from tensorflow.python.ops import image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops

import tensorflow as tf


# In[2]:

def crop_to_bounding_box(image, offset_height, offset_width, target_height,
                         target_width, dynamic_shape=False):
  """Crops an image to a specified bounding box.

  This op cuts a rectangular part out of `image`. The top-left corner of the
  returned image is at `offset_height, offset_width` in `image`, and its
  lower-right corner is at
  `offset_height + target_height, offset_width + target_width`.

  Args:
    image: 3-D tensor with shape `[height, width, channels]`
    offset_height: Vertical coordinate of the top-left corner of the result in
                   the input.
    offset_width: Horizontal coordinate of the top-left corner of the result in
                  the input.
    target_height: Height of the result.
    target_width: Width of the result.
    dynamic_shape: Whether the input image has undertermined shape. If set to
      `True`, shape information will be retrieved at run time. Default to
      `False`.

  Returns:
    3-D tensor of image with shape `[target_height, target_width, channels]`

  Raises:
    ValueError: If the shape of `image` is incompatible with the `offset_*` or
    `target_*` arguments, and `dynamic_shape` is set to `False`.
  """
  image = tf.convert_to_tensor(image, name='image')
  _Check3DImage(image, require_static=(not dynamic_shape))
  height, width, _ = _ImageDimensions(image, dynamic_shape=dynamic_shape)

  if not dynamic_shape:
    if offset_width < 0:
      raise ValueError('offset_width must be >= 0.')
    if offset_height < 0:
      raise ValueError('offset_height must be >= 0.')

    if width < (target_width + offset_width):
      raise ValueError('width must be >= target + offset.')
    if height < (target_height + offset_height):
      raise ValueError('height must be >= target + offset.')

  cropped = array_ops.slice(image,
                            array_ops.pack([offset_height, offset_width, 0]),
                            array_ops.pack([target_height, target_width, -1]))

  return cropped


# In[3]:

def pad_to_bounding_box(image, offset_height, offset_width, target_height,
                        target_width, dynamic_shape=False):
  """Pad `image` with zeros to the specified `height` and `width`.

  Adds `offset_height` rows of zeros on top, `offset_width` columns of
  zeros on the left, and then pads the image on the bottom and right
  with zeros until it has dimensions `target_height`, `target_width`.

  This op does nothing if `offset_*` is zero and the image already has size
  `target_height` by `target_width`.

  Args:
    image: 3-D tensor with shape `[height, width, channels]`
    offset_height: Number of rows of zeros to add on top.
    offset_width: Number of columns of zeros to add on the left.
    target_height: Height of output image.
    target_width: Width of output image.
    dynamic_shape: Whether the input image has undertermined shape. If set to
      `True`, shape information will be retrieved at run time. Default to
      `False`.

  Returns:
    3-D tensor of shape `[target_height, target_width, channels]`
  Raises:
    ValueError: If the shape of `image` is incompatible with the `offset_*` or
      `target_*` arguments, and `dynamic_shape` is set to `False`.
  """
  image = tf.convert_to_tensor(image, name='image')
  _Check3DImage(image, require_static=(not dynamic_shape))
  height, width, depth = _ImageDimensions(image, dynamic_shape=dynamic_shape)

  after_padding_width = target_width - offset_width - width
  after_padding_height = target_height - offset_height - height

  if not dynamic_shape:
    if target_width < width:
      raise ValueError('target_width must be >= width')
    if target_height < height:
      raise ValueError('target_height must be >= height')

    if after_padding_width < 0:
      raise ValueError('target_width not possible given '
                       'offset_width and image width')
    if after_padding_height < 0:
      raise ValueError('target_height not possible given '
                       'offset_height and image height')

  # Do not pad on the depth dimensions.
  if (dynamic_shape or offset_width or offset_height or
      after_padding_width or after_padding_height):
    paddings = array_ops.reshape(
      array_ops.pack([offset_height, after_padding_height,
                      offset_width, after_padding_width,
                      0, 0]),
      [3, 2])
    padded = array_ops.pad(image, paddings)
    if not dynamic_shape:
      padded.set_shape([target_height, target_width, depth])
  else:
    padded = image

  return padded


# In[4]:

def resize_image_with_crop_or_pad(image, target_height, target_width,
                                  dynamic_shape=False):
  """Crops and/or pads an image to a target width and height.

  Resizes an image to a target width and height by either centrally
  cropping the image or padding it evenly with zeros.

  If `width` or `height` is greater than the specified `target_width` or
  `target_height` respectively, this op centrally crops along that dimension.
  If `width` or `height` is smaller than the specified `target_width` or
  `target_height` respectively, this op centrally pads with 0 along that
  dimension.

  Args:
    image: 3-D tensor of shape [height, width, channels]
    target_height: Target height.
    target_width: Target width.
    dynamic_shape: Whether the input image has undertermined shape. If set to
      `True`, shape information will be retrieved at run time. Default to
      `False`.

  Raises:
    ValueError: if `target_height` or `target_width` are zero or negative.

  Returns:
    Cropped and/or padded image of shape
    `[target_height, target_width, channels]`
  """
  image = tf.convert_to_tensor(image, name='image')
  _Check3DImage(image, require_static=(not dynamic_shape))
  original_height, original_width, _ =     _ImageDimensions(image, dynamic_shape=dynamic_shape)

  if target_width <= 0:
    raise ValueError('target_width must be > 0.')
  if target_height <= 0:
    raise ValueError('target_height must be > 0.')

  if dynamic_shape:
    max_ = math_ops.maximum
    min_ = math_ops.minimum
  else:
    max_ = max
    min_ = min

  width_diff = target_width - original_width
  offset_crop_width = max_(-width_diff // 2, 0)
  offset_pad_width = max_(width_diff // 2, 0)

  height_diff = target_height - original_height
  offset_crop_height = max_(-height_diff // 2, 0)
  offset_pad_height = max_(height_diff // 2, 0)

  # Maybe crop if needed.
  cropped = crop_to_bounding_box(image, offset_crop_height, offset_crop_width,
                                 min_(target_height, original_height),
                                 min_(target_width, original_width),
                                 dynamic_shape=dynamic_shape)

  # Maybe pad if needed.
  resized = pad_to_bounding_box(cropped, offset_pad_height, offset_pad_width,
                                target_height, target_width,
                                dynamic_shape=dynamic_shape)

  if resized.get_shape().ndims is None:
    raise ValueError('resized contains no shape.')
  if not resized.get_shape()[0].is_compatible_with(target_height):
    raise ValueError('resized height is not correct.')
  if not resized.get_shape()[1].is_compatible_with(target_width):
    raise ValueError('resized width is not correct.')
  return resized


# In[5]:

def _ImageDimensions(images, dynamic_shape=False):
  """Returns the dimensions of an image tensor.
  Args:
    images: 4-D Tensor of shape [batch, height, width, channels]
    dynamic_shape: Whether the input image has undertermined shape. If set to
      `True`, shape information will be retrieved at run time. Default to
      `False`.

  Returns:
    list of integers [batch, height, width, channels]
  """
  # A simple abstraction to provide names for each dimension. This abstraction
  # should make it simpler to switch dimensions in the future (e.g. if we ever
  # want to switch height and width.)
  if dynamic_shape:
    return array_ops.unpack(array_ops.shape(images))
  else:
    return images.get_shape().as_list()


# In[6]:

def _Check3DImage(image, require_static=True):
  """Assert that we are working with properly shaped image.
  Args:
    image: 3-D Tensor of shape [height, width, channels]
    require_static: If `True`, requires that all dimensions of `image` are
      known and non-zero.

  Raises:
    ValueError: if image.shape is not a [3] vector.
  """
  try:
    image_shape = image.get_shape().with_rank(3)
  except ValueError:
    raise ValueError('\'image\' must be three-dimensional.')
  if require_static and not image_shape.is_fully_defined():
    raise ValueError('\'image\' must be fully defined.')
  if any(x == 0 for x in image_shape):
    raise ValueError('all dims of \'image.shape\' must be > 0: %s' %
                     image_shape)


# In[7]:

image_ops.resize_image_with_crop_or_pad = resize_image_with_crop_or_pad
#image_ops.crop_to_bounding_box = crop_to_bounding_box
#image_ops.pad_to_bounding_box = pad_to_bounding_box
#image_ops._ImageDimensions = _ImageDimensions
#image_ops._Check3DImage = _Check3DImage

