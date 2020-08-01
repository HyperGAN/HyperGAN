import PIL
import torch.nn.functional as F
import torch
from collections.abc import Sequence, Iterable
import warnings

from torchvision.transforms import functional as F


class CropResizeTransform(object):
    """Crop the image to min(h,w) then resize it

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=PIL.Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        width, height = img.size

        min_wh = min(width, height)
        min_w = width * self.size[0]/self.size[1]
        min_h = height * self.size[1]/self.size[0]
        min_size = [min_w, min_h]
        img = F.center_crop(img, min_size)

        return F.resize(img, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)

