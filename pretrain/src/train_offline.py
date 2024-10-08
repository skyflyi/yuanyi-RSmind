import os
import argparse
from mindspore import context
from mindspore.train.model import Model
from mindspore.context import ParallelMode
import mindspore.nn as nn
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import LossMonitor, TimeMonitor, SummaryCollector
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.common import set_seed
from src import dataset_train as data_generator
from src import loss, learning_rates, callback
from src.metrics import Middle_loss
from Swin_MAE_BYOL import MAE_BYOL
import mindspore.ops as ops
import time
set_seed(1)


# ubuntu18.04  GPU CUDA 10.1 MindSpore 1.3.0 python 3.7.5
# 基于VOC公开数据集的本地训练
# 针对SAR图像，禁用数据增强

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
class BuildTrainNetwork(nn.Cell):
    def __init__(self, network, loss_):
        super(BuildTrainNetwork, self).__init__()
        self.network = network
        self.loss_ = loss_

    # def construct(self, x_online):
    #     online_rec, mask = self.network(x_online)
    #     net_loss = self.loss_(x_online, online_rec, mask)
    #     return net_loss

    def construct(self, x_online, x_target, mask):
        online_x1, online_x2, x_rec, target_x1, target_x2, x_online, mask = self.network(x_online, x_target, mask)
        net_loss = self.loss_(online_x1, online_x2, x_rec, target_x1, target_x2, x_online, mask)
        # out = self.network(x_online, x_target, mask)
        # net_loss = self.loss_(out=[online_contrast, online_rec, target_contrast], label=[x_online, mask])
        return net_loss

class CustomTrainOneStepCell(nn.Cell):
    """自定义训练网络"""
    def __init__(self, network, optimizer, sens=1.0):
        """入参有三个：训练网络，优化器和反向传播缩放比例"""
        super(CustomTrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network                    # 定义前向网络
        self.network.set_grad()                   # 构建反向网络
        self.optimizer = optimizer                # 定义优化器
        self.weights = self.optimizer.parameters  # 待更新参数
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)  # 反向传播获取梯度

    def construct(self, x_online, x_target):
        loss = self.network(x_online, x_target)                    # 执行前向网络，计算当前输入的损失函数值
        print(loss)
        grads = self.grad(self.network, self.weights)(x_online, x_target, loss)  # 进行反向传播，计算梯度
        self.optimizer(grads)# 使用优化器更新梯度
        return loss


def parse_args():
    parser = argparse.ArgumentParser('MindSpore MAE_BYOL training')
    # dataset
    parser.add_argument('--batch_size', type=int, default=4, help='batch size, default=32 ')
    parser.add_argument('--image_size', type=int, default=512, help='crop size')
    parser.add_argument('--image_mean', type=list, default=[86.6077, 89.0429, 71.817],
                        help='image mean')
    parser.add_argument('--image_std', type=list, default=[28.0801, 27.0622, 30.239],
                        help='image std')
    # optimizer
    parser.add_argument('--train_epochs', type=int, default=100, help='epoch, default=300')
    parser.add_argument('--save_epochs', type=int, default=2, help='steps interval for saving')
    parser.add_argument('--save_steps', type=int, default=100, help='steps interval for saving')
    parser.add_argument('--lr_type', type=str, default='poly', choices=['poly', 'step', 'cos'], help='type of learning rate, default = cos')
    parser.add_argument('--base_lr', type=float, default=0.01, help='base learning rate cos:0.015 poly:0.007')#0.07
    parser.add_argument('--end_lr', type=float, default=0.0005, help='base learning rate cos:0.015 poly:0.007')  # 0.07
    parser.add_argument('--lr_decay_epoch', type=int, default=5, help='learning rate decay step')
    parser.add_argument('--lr_decay_rate', type=float, default=0.01, help='learning rate decay rate')

    # train
    parser.add_argument('--rank', type=int, default=0, help='local rank of distributed')
    parser.add_argument('--group_size', type=int, default=1, help='world size of distributed')

    # ModelArts
    parser.add_argument('--train_url', type=str, default='/home/hw/shijie/MAE_BYOL/pretrain_out',
                        help='where training log and CKPTs saved')
    parser.add_argument('--data_url', type=str, default='/home/hw/shijie/MAE_BYOL/data/',
                        help='the directory path of saved file')
    parser.add_argument('--train_data_filename', type=str, default='pretrain_mask04.mindrecord', help='Name of the MindRecord file')

    parser.add_argument('--loss_scale', type=float, default=4800.0, help='loss scale')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='loss scale  default = 0.0001 or 5e-4')
    parser.add_argument('--keep_checkpoint_max', type=int, default=500, help='max checkpoint for saving default=200')
    args, _ = parser.parse_known_args()
    return args


def train():
    args = parse_args()
    context.set_context(mode=context.PYNATIVE_MODE, enable_auto_mixed_precision=True, save_graphs=False,
                            device_target="Ascend", device_id=int(os.getenv("DEVICE_ID")))

    # mox.file.set_auth(ak='3YUHYFFBN3WOUMN3PMUD', sk='w4eWQdkT22GPthPcTJulTrpSKxtkeJKeAXJGSzpe',
    #                   server='obs.cn-northwest-229.yantachaosuanzhongxin.com')
    init()
    args.rank = get_rank()
    args.group_size = get_group_size()
    print("rank_id is {}, rank_size is {}".format(args.rank, args.group_size))
    context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True, device_num=args.group_size)

    local_data_url = args.data_url
    # local_train_url = args.train_url
    train_data_file = os.path.join(local_data_url, args.train_data_filename)

    print(args)
    dataset = data_generator.SegDataset(data_file=train_data_file, batch_size=args.batch_size, image_size=args.image_size,
                                              image_mean=args.image_mean, image_std=args.image_std, num_readers=None,
                                              num_parallel_calls=None, shard_id=args.rank, shard_num=args.group_size)
    train_dataset = dataset.get_dataset(repeat=1)
    eval_dataset = dataset.get_dataset_eval(repeat=1)

    iters_per_epoch = train_dataset.get_dataset_size()
    print("iters_per_epoch = ", iters_per_epoch)

    total_train_steps = iters_per_epoch * args.train_epochs
    decay_train_steps = iters_per_epoch * args.lr_decay_epoch
    if args.lr_type == 'cos':
        lr_iter = learning_rates.cosine_lr(args.base_lr, total_train_steps, total_train_steps)
    elif args.lr_type == 'poly':
        lr_iter = learning_rates.poly_lr(args.base_lr, decay_steps=decay_train_steps, total_steps=total_train_steps, end_lr=args.end_lr, power=0.9)
    elif args.lr_type == 'exp':
        lr_iter = learning_rates.exponential_lr(args.base_lr, args.lr_decay_step, args.lr_decay_rate,
                                                total_train_steps, staircase=True)
    else:
        raise ValueError('unknown learning rate type')

    network = MAE_BYOL(batch_size=args.batch_size)

    loss_ = loss.MAE_BYOL_loss()
    # loss_.add_flags_recursive(fp32=True)
    train_net = BuildTrainNetwork(network, loss_)
    #损失超过界限依旧更新参数(损失策略)
    manager_loss_scale = FixedLossScaleManager(args.loss_scale, drop_overflow_update=False)
    # opt = nn.Adam(params=train_net.trainable_params(), learning_rate=args.base_lr)
                      #                   weight_decay=args.weight_decay, loss_scale=args.loss_scale)
    opt = nn.Momentum(params=train_net.trainable_params(), learning_rate=lr_iter, momentum=0.9,
                      weight_decay=args.weight_decay, loss_scale=args.loss_scale)
    # train_net = CustomTrainOneStepCell(network, opt)
    #运行过程中精度策略
    amp_level = "O3"

    #metrics:运行过程中统计的参数
    model = Model(network=train_net, loss_fn=None, optimizer=opt, amp_level=amp_level, loss_scale_manager=manager_loss_scale,
                  eval_network=network, metrics={"loss_middle": Middle_loss(smooth=1e-5)})

    # callback for saving ckpts
    time_cb = TimeMonitor()
    loss_cb = LossMonitor(per_print_times=300)
    cbs = [time_cb, loss_cb]

    #在第一张卡上保存模型与参数
    # eval_out = {"epoch": [], "Miou": [], 'OA': [], 'Precision': [], 'Recall': [], 'Kappa': [], 'F1scores': []}
    # path_curve = local_train_url + '/curve'
    # if not os.path.exists(path_curve):
    #     os.makedirs(path_curve)
    local_train_url = os.getcwd()
    if args.rank == 0:
        with open(os.path.join(args.train_url, "all_parameters_dic.txt"), mode='w') as f:
            f.write(str(network.parameters_dict()))
        ckpoint_config = CheckpointConfig(save_checkpoint_steps=args.save_epochs * iters_per_epoch, keep_checkpoint_max=args.keep_checkpoint_max)
        ckpoint_cb = ModelCheckpoint(prefix='MAE_BYOL_SwinTransformer', directory=local_train_url, config=ckpoint_config)
        cbs.append(ckpoint_cb)
    train_loss = {'Loss': [], 'step': []}
    middle_loss = {'MAE_loss': [], 'BYOL_loss': [], 'step': []}
    eval_cb = callback.EvalCallBack(model, args, train_loss, middle_loss, eval_dataset, local_train_url, iters_per_epoch)
    cbs.append(eval_cb)

    #验证,记录准确率
    # eval_out = {"epoch": [], "Miou": [], 'OA': [], 'tr_OA':[], 'Precision': [], 'Recall': []}
    # eval_cb = callback.EvalCallBack(model, val_dataset, tr_dataset, args.eval_per_epoch, eval_out)

    model.train(args.train_epochs, train_dataset, callbacks=cbs)

    # with open(local_train_url + '/Record_train_parameters_set.txt', 'a') as f:
    #     f.write('\n\n')
    #     f.write('prepare_time:                       ' + str(time2 - time1) + 's\n')
    #     f.write('load_dataset_time:                  ' + str(time3 - time2) + 's\n')
    #     f.write('initial_network_time:               ' + str(time4 - time3) + 's\n')
    #     f.write('train_time:                         ' + str((time5 - time4)/60) + 'mins\n')
    #     f.write('loss:                               ' + str(loss_cb))
    #     # f.write('copy_time:                          ' + str(time6 - time5) + 's\n')
    # with open(os.path.join(local_train_url, 'log_dice.txt'), 'w') as f:
    #     f.write(str(eval_out))
    #     f.write('\n')
    #     f.write(str(train_loss))
    # print(train_loss)
    # callback.eval_show(eval_out, train_loss, args.group_size, local_train_url, args.num_classes)

    # if args.modelArts_mode:
    #     # copy train result from cache to obs
    #     if args.rank == 0:
    #         mox.file.copy_parallel(src_url=local_train_url, dst_url=args.train_url)
            # mox.file.copy_parallel(src_url=local_plog_path, dst_url=args.train_url)

if __name__ == '__main__':
    # 记录文件，时间和评估指标
    args = parse_args()

    time_start = time.time()

    train()

    time_end = time.time()
    print(time_end - time_start)
