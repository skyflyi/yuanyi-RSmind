from mindspore import Tensor
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter
import numpy as np

def compute_loss(online_x1, online_x2, x_rec, target_x1, target_x2, x_online, mask):
    patch_size = 4
    in_channels = 3
    B = mask.shape[0]
    loss_rec = np.abs(x_rec - x_online)
    mask = mask.repeat(patch_size, axis=1).repeat(patch_size, axis=2)
    mask_token = np.expand_dims(mask, axis=1)
    mask_token = np.broadcast_to(mask_token, (B, in_channels, mask_token.shape[2], mask_token.shape[3]))
    mask_token = mask_token * 0.01
    loss_rec = loss_rec * mask_token
    loss_rec = loss_rec.sum() / mask_token.sum()

    loss_contrast_1 = online_x1 * target_x2
    loss_contrast_1 = loss_contrast_1.sum(axis=-1)
    loss_contrast_1 = (2 - 2 * loss_contrast_1).astype(np.float32)
    loss_contrast_1 = loss_contrast_1.sum(axis=0) / B

    loss_contrast_2 = online_x2 * target_x1
    loss_contrast_2 = loss_contrast_2.sum(axis=-1)
    loss_contrast_2 = (2 - 2 * loss_contrast_2).astype(np.float32)
    loss_contrast_2 = loss_contrast_2.sum(axis=0) / B
    loss_contrast = (loss_contrast_1 + loss_contrast_2) / 2.

    return np.array([loss_rec, loss_contrast])