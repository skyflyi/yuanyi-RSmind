import mindspore
from mindspore import Tensor, ms_function
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter
from mindspore.ops import operations as P
import numpy as np


class MAE_BYOL_loss(nn.Cell):
    def __init__(self, patch_size=4, in_channels=3, feature_dim=768, gamma=0.5):
        super(MAE_BYOL_loss, self).__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.gamma = gamma
        # self.step = Parameter(Tensor(1.0), requires_grad=False)
        self.cast = ops.Cast()
        # self.scale = scale

    @ms_function()
    def construct(self, online_x1, online_x2, x_rec, target_x1, target_x2, x_online, mask):
        B,L = mask.shape[0], mask.shape[1]
        loss_rec = ops.abs(x_rec - x_online)
        # # mask_token = mask.squeeze()
        # mask_token = mask_token.reshape((B, int(L**0.5), int(L**0.5)))
        # # mask_token = mask_token.repeat(self.patch_size, axis=1).repeat(self.patch_size, axis=2)

        # mask = mask.repeat(self.patch_size, axis=1).repeat(self.patch_size, axis=2)
        # mask_token = mask.expand_dims(axis=1)
        # mask_token = ops.function.broadcast_to(mask_token, (-1, self.in_channels, -1, -1))
        # mask_token = mask_token*0.01
        mask_token = (ops.function.broadcast_to((mask.repeat(self.patch_size, axis=1).repeat(self.patch_size, axis=2)).expand_dims(axis=1), (-1, self.in_channels, -1, -1))) * 0.01
        # # loss_rec = loss_rec.astype(mstype.float32)
        # # mask_token = mask_token.astype(mstype.float32)
        # mask_token = (mask_token/self.scale).astype(mstype.float16)
        # # if (self.step % 5 == 0):
        # #     print(f"loss_rec:{loss_rec.sum()}")
        # #     print(f"mask_token:{mask_token.sum()}")
        # loss_rec = ops.mul(loss_rec, mask_token)
        # loss_rec = loss_rec.sum() / mask_token.sum()
        loss_rec = (ops.mul(loss_rec, mask_token)).sum() / mask_token.sum()

        # loss_contrast_1 = ops.mul(online_x1, target_x2)
        # loss_contrast_1 = loss_contrast_1.sum(axis=-1)
        # loss_contrast_1 = (2 - 2*loss_contrast_1).astype(mstype.float32)
        # loss_contrast_1 = loss_contrast_1.sum(axis=0)/B
        loss_contrast_1 = ((2 - 2*(ops.mul(online_x1, target_x2)).sum(axis=-1)).astype(mstype.float32)).sum(axis=0) / B

        # loss_contrast_2 = ops.mul(online_x2, target_x1)
        # loss_contrast_2 = loss_contrast_2.sum(axis=-1)
        # loss_contrast_2 = (2 - 2 * loss_contrast_2).astype(mstype.float32)
        # loss_contrast_2 = loss_contrast_2.sum(axis=0) / B
        loss_contrast_2 = ((2 - 2 * (ops.mul(online_x2, target_x1)).sum(axis=-1)).astype(mstype.float32)).sum(
            axis=0) / B

        loss_contrast = (loss_contrast_1+loss_contrast_2) / 2.

        loss = self.gamma*loss_rec + (1. - self.gamma)*loss_contrast
        # if(self.step%100==0):
        #     print(f"loss_rec:{loss_rec}")
        #     print(f"loss_contrast:{loss_contrast}")
        # self.step = self.step + 1.0
        print(f"loss_rec:{loss_rec}")
        print(f"loss_contrast:{loss_contrast}")
        return loss

    # def construct(self, out, label):
    #     online_contrast, online_rec, target_contrast = out[0], out[1], out[2]
    #     x_online, mask = label[0],label[1]
    #     B,L = mask.shape[0], mask.shape[1]
    #     loss_rec = ops.abs(online_rec - x_online)
    #     # # mask_token = mask.squeeze()
    #     # mask_token = mask_token.reshape((B, int(L**0.5), int(L**0.5)))
    #     # # mask_token = mask_token.repeat(self.patch_size, axis=1).repeat(self.patch_size, axis=2)
    #     mask = mask.repeat(self.patch_size, axis=1).repeat(self.patch_size, axis=2)
    #     mask_token = mask.expand_dims(axis=1)
    #     mask_token = ops.function.broadcast_to(mask_token, (-1, self.in_channels, -1, -1))
    #     # mask_token = self.cast(mask_token, mstype.float32)
    #     mask_token = mask_token*0.01
    #     # # loss_rec = loss_rec.astype(mstype.float32)
    #     # # mask_token = mask_token.astype(mstype.float32)
    #     # mask_token = (mask_token/self.scale).astype(mstype.float16)
    #     # # if (self.step % 5 == 0):
    #     # #     print(f"loss_rec:{loss_rec.sum()}")
    #     # #     print(f"mask_token:{mask_token.sum()}")
    #     loss_rec = ops.mul(loss_rec, mask_token)
    #     # loss_rec = self.cast(loss_rec, mstype.float32)
    #     loss_rec = loss_rec.sum() / mask_token.sum()
    #
    #     loss_contrast = ops.mul(online_contrast, target_contrast)
    #     loss_contrast = loss_contrast.sum(axis=-1)
    #     loss_contrast = (2 - 2*loss_contrast).astype(mstype.float32)
    #     loss_contrast = loss_contrast.sum(axis=0)/B
    #
    #     loss = self.gamma*loss_rec + (1. - self.gamma)*loss_contrast
    #     if(self.step%100==0):
    #         print(f"loss_rec:{loss_rec}")
    #         print(f"loss_contrast:{loss_contrast}")
    #     self.step = self.step + 1.0
    #     return loss



    # def construct(self, x_online, online_rec, mask):
    #     B,L = mask.shape[0], mask.shape[1]
    #     # mean = x_online.mean(axis=1, keep_dims=True)
    #     # std = x_online.std(axis=1, keepdims=True)
    #     # x_online_norm = (x_online - mean) / std
    #     # print(x_online[0, :, 0:3, 0:3])
    #     # print(online_rec[0, :, 0:3, 0:3])
    #
    #
    #     loss_rec =ops.abs(online_rec - x_online).astype(mstype.float32)
    #     # print((loss_rec)[0, 0, 0:6, 0:6])
    #     mask = mask.squeeze()
    #     mask = mask.reshape((B, int(L**0.5), int(L**0.5)))
    #     mask = mask.repeat(self.patch_size, axis=1).repeat(self.patch_size, axis=2)
    #     # print(mask.sum())
    #     # mask = mask[0, :, :]
    #     mask = mask.expand_dims(axis=1)
    #     mask = ops.function.broadcast_to(mask, (B, self.in_channels, -1, -1))
    #     mask = mask.astype(loss_rec.dtype)
    #     # print((loss_rec * mask_token)[0, 0, 0:6, 0:6])
    #     # print(np.sum(loss_rec * mask_token))
    #     rec_sum = loss_rec.sum()
    #     mask_sum = mask.sum()
    #     rec_mask_sum = (loss_rec * mask).sum()
    #     loss_rec = ((loss_rec * mask).sum()) / mask.sum()
    #     return loss_rec

    # def construct(self, x_online, online_rec, mask):
    #     B = x_online.shape[0]
    #     mean = x_online.mean(axis=1, keep_dims=True)
    #     std = x_online.std(axis=1, keepdims=True)
    #     x_online_norm = (x_online - mean) / std
    #
    #     loss_rec = ops.abs(online_rec - x_online_norm)
    #     mask_token = mask.repeat(self.patch_size, axis=0).repeat(self.patch_size, axis=1)
    #     mask_token = ops.broadcast_to(Tensor(mask_token), (B, self.in_channels, -1, -1))
    #     mask_token = mask_token.astype(loss_rec.dtype)
    #     loss_rec = ((loss_rec * mask_token).sum()) / mask_token.sum()
    #     return loss_rec

    # def construct(self, online_contrast, target_contrast):
    #     online_contrast = self.norm(online_contrast)
    #     target_contrast = self.norm(target_contrast)
    #     loss_contrast = ops.mul(online_contrast, target_contrast)
    #     loss_contrast = loss_contrast.sum(axis=-1)
    #     loss_contrast = 2 - 2*loss_contrast
    #     loss_contrast = loss_contrast.sum(axis=0)
    #     return loss_contrast
class MAE_loss(nn.Cell):
    def __init__(self, patch_size=4, in_channels=3, feature_dim=768, gamma=0.5):
        super(MAE_loss, self).__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.gamma = gamma
        self.step = Parameter(Tensor(1.0), requires_grad=False)
        self.cast = ops.Cast()
        # self.scale = scale
    def construct(self, x_rec, x_online, mask):
        # loss_rec = ops.abs(x_rec - x_online)
        # # mask_token = mask.squeeze()
        # mask_token = mask_token.reshape((B, int(L**0.5), int(L**0.5)))
        # # mask_token = mask_token.repeat(self.patch_size, axis=1).repeat(self.patch_size, axis=2)

        # mask = mask.repeat(self.patch_size, axis=1).repeat(self.patch_size, axis=2)
        # mask_token = mask.expand_dims(axis=1)
        # mask_token = ops.function.broadcast_to(mask_token, (-1, self.in_channels, -1, -1))
        # mask_token = mask_token*0.01
        mask_token = ((mask.repeat(self.patch_size, axis=1).repeat(self.patch_size, axis=2)).expand_dims(axis=1)) * 0.01
        # # loss_rec = loss_rec.astype(mstype.float32)
        # # mask_token = mask_token.astype(mstype.float32)
        # mask_token = (mask_token/self.scale).astype(mstype.float16)
        # # if (self.step % 5 == 0):
        # #     print(f"loss_rec:{loss_rec.sum()}")
        # #     print(f"mask_token:{mask_token.sum()}")
        # loss_rec = ops.mul(loss_rec, mask_token)
        # loss_rec = loss_rec.sum() / mask_token.sum()
        loss_rec = ops.l1_loss(x_rec, x_online, reduction='none')
        loss_rec = (ops.mul(loss_rec, mask_token)).sum() / mask_token.sum() / self.in_channels

        # if(self.step%10==0):
        #     print(f"loss_rec:{loss_rec[0]}")
        # self.step = self.step + 1.0
        return loss_rec
