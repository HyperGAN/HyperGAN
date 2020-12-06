#adapted from https://raw.githubusercontent.com/PeterouZh/Omni-GAN-PyTorch/main/exp/omni_loss/omni_loss.py
import os
import torch
import torch.nn.functional as F
import random
from hypergan.losses.base_loss import BaseLoss


def multilabel_categorical_crossentropy(y_true, y_pred, margin=0., gamma=1.):
  """
  y_true: positive=1, negative=0, ignore=-1

  """
  y_true = y_true.clamp(-1, 1)
  if len(y_pred.shape) > 2:
    y_true = y_true.view(y_true.shape[0], 1, 1, -1)
    _, _, h, w = y_pred.shape
    y_true = y_true.expand(-1, h, w, -1)
    y_pred = y_pred.permute(0, 2, 3, 1)

  y_pred = y_pred + margin
  y_pred = y_pred * gamma

  y_pred[y_true == 1] = -1 * y_pred[y_true == 1]
  y_pred[y_true == -1] = -1e12

  y_pred_neg = y_pred.clone()
  y_pred_neg[y_true == 1] = -1e12

  y_pred_pos = y_pred.clone()
  y_pred_pos[y_true == 0] = -1e12

  zeros = torch.zeros_like(y_pred[..., :1])
  y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
  y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
  neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
  pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
  return neg_loss + pos_loss


class OmniLossInternal(object):

  def __init__(self, default_label=0, margin=0., gamma=1.):
    self.default_label = default_label
    self.margin = margin
    self.gamma = gamma
    pass

  @staticmethod
  def get_one_hot(label_list, one_hot, b, filled_value=0):

    for label in label_list:
      if isinstance(label, int):
        label = torch.empty(b, dtype=torch.int64, device=one_hot.device).fill_(label)
      one_hot.scatter_(dim=1, index=label.view(-1, 1), value=filled_value)
    return one_hot

  def __call__(self, pred, positive=None, negative=None, default_label=None, margin=None, gamma=None):
    default_label = self.default_label if default_label is None else default_label
    margin = self.margin if margin is None else margin
    gamma = self.gamma if gamma is None else gamma

    b, nc = pred.shape[:2]
    label_onehot = torch.empty(b, nc, dtype=torch.int64, device=pred.device).fill_(default_label)

    if positive is not None:
      label_onehot = OmniLossInternal.get_one_hot(label_list=positive, one_hot=label_onehot, b=b, filled_value=1)

    if negative is not None:
      label_onehot = OmniLossInternal.get_one_hot(label_list=negative, one_hot=label_onehot, b=b, filled_value=0)

    loss = multilabel_categorical_crossentropy(
      y_true=label_onehot, y_pred=pred, margin=margin, gamma=gamma)
    loss_mean = loss.mean()
    return loss_mean

class OmniLoss(BaseLoss):
    def _forward(self, d_real, d_fake):
        self.omni_loss_internal = OmniLossInternal()
        classes = 2
        y_ = self.gan.latent_y
        p = self.gan.classification

        d_real_positive = (self.gan.classification, classes)
        d_loss_real = self.omni_loss_internal(pred=d_real, positive=d_real_positive, default_label=0)
        d_fake_positive = (classes + 1,)
        d_loss_fake = self.omni_loss_internal(pred=d_fake, positive=d_fake_positive, default_label=0)
        g_fake_positive = (y_, classes)
        g_loss = self.omni_loss_internal(pred=d_fake, positive=g_fake_positive, default_label=0)
        d_loss = d_loss_fake + d_loss_real

        return [d_loss, g_loss]
