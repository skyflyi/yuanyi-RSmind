from mindspore.train.callback import Callback
import matplotlib.pyplot as plt
import os
from mindspore import save_checkpoint
import numpy as np
class EvalCallBack(Callback):
    """Precision verification using callback function."""
    def __init__(self, models, args, train_loss, middle_loss, eval_dataset, local_train_url, iters_per_epoch):
        super(EvalCallBack, self).__init__()
        self.models = models
        self.save_steps = args.save_steps
        self.train_loss = train_loss
        self.middle_loss = middle_loss
        self.eval_dataset = eval_dataset
        self.local_train_url = local_train_url
        self.iters_per_epoch = iters_per_epoch

    def step_end(self, run_context):
        cb_param = run_context.original_args()
        cur_step = cb_param.cur_step_num
        cur_epoch = cb_param.cur_epoch_num
        los = cb_param.net_outputs.asnumpy()
        if((cur_step-1) % self.save_steps==0):
            self.train_loss['Loss'].append(los)
            self.train_loss['step'].append(cur_step)

            loss_middle = self.models.eval(self.eval_dataset)
            self.middle_loss['MAE_loss'].append(loss_middle['loss_middle'][0])
            self.middle_loss['BYOL_loss'].append(loss_middle['loss_middle'][1])
            self.middle_loss['step'].append(cur_step)
            # print(self.train_loss)
            plt.figure(1)
            plt.xlabel("Step number")
            plt.ylabel("Loss")
            plt.title("variation chart of Training Loss")
            plt.plot(self.train_loss["step"], self.train_loss["Loss"], "black", label='train_loss')
            plt.savefig(os.path.join(self.local_train_url, "Curve_of_loss_change.png"))

            plt.figure(2)
            plt.xlabel("Step number")
            plt.ylabel("MAE loss")
            plt.title("variation chart of Training MAE Loss")
            plt.plot(self.middle_loss["step"], self.middle_loss["MAE_loss"], "black", label='train_MAE_loss')
            plt.savefig(os.path.join(self.local_train_url, "Curve_of_MAE_loss_change.png"))

            plt.figure(3)
            plt.xlabel("Step number")
            plt.ylabel("BYOL Loss")
            plt.title("variation chart of Training BYOL Loss")
            plt.plot(self.middle_loss["step"], self.middle_loss["BYOL_loss"], "black", label='train_BYOL_loss')
            plt.savefig(os.path.join(self.local_train_url, "Curve_of_BYOL_loss_change.png"))

            with open(os.path.join(self.local_train_url, 'log_dice.txt'), 'w') as f:
                f.write(str(self.train_loss))
                f.write('\n')
                f.write(str(self.middle_loss))

class TrainCallBack(Callback):
    """Precision verification using callback function."""
    def __init__(self, model, args, train_loss, local_train_url, iters_per_epoch):
        super(TrainCallBack, self).__init__()
        self.models = model
        self.save_steps = args.save_steps
        self.train_loss = train_loss
        self.local_train_url = local_train_url
        self.iters_per_epoch = iters_per_epoch

    def step_end(self, run_context):
        cb_param = run_context.original_args()
        cur_step = cb_param.cur_step_num
        cur_epoch = cb_param.cur_epoch_num
        los = cb_param.net_outputs.asnumpy()
        if((cur_step) % self.iters_per_epoch==0):
            self.train_loss['Loss'].append(los)
            self.train_loss['step'].append(cur_step)
            # print(self.train_loss)
            plt.figure(1)
            plt.xlabel("Step number")
            plt.ylabel("Loss")
            plt.title("variation chart of Training Loss")
            plt.plot(self.train_loss["step"], self.train_loss["Loss"], "black", label='train_loss')
            plt.savefig(os.path.join(self.local_train_url, "Curve_of_loss_change.png"))
            with open(os.path.join(self.local_train_url, 'log_dice.txt'), 'w') as f:
                f.write(str(self.train_loss))

def eval_show(epoch_per_eval, train_loss, group_size, path, num_classes):
    plt.figure(1)
    plt.xlabel("Epoch number")
    plt.ylabel("Model accuracy: OA")
    plt.title("Model accuracy OA variation chart")
    plt.plot(epoch_per_eval["epoch"], epoch_per_eval["OA"], "green", label='val')
    plt.legend()
    plt.savefig(os.path.join(path, "Curve_of_OA_change.png"))

    # loss_len = len(train_loss)
    # singe_loss = []
    # m = group_size-1
    # while m < loss_len:
    #     singe_loss.append(train_loss[m])
    #     m = m + group_size
    plt.figure(2)
    plt.xlabel("Epoch number")
    plt.ylabel("Loss")
    plt.title("variation chart of Training Loss")
    plt.plot(epoch_per_eval["epoch"], train_loss["Loss"], "black", label='train_loss')
    plt.legend()
    plt.savefig(os.path.join(path, "Curve_of_loss_change.png"))

    pre = epoch_per_eval["Precision"]
    re = epoch_per_eval["Recall"]
    epoch_num = len(epoch_per_eval["epoch"])
    seq_pre = np.zeros((num_classes, epoch_num))
    seq_re = np.zeros((num_classes, epoch_num))
    for i in range(num_classes):
        for j in range(epoch_num):
            seq_pre[i][j] = pre[j][i]
    for i in range(num_classes):
        for j in range(epoch_num):
            seq_re[i][j] = re[j][i]

    class_label = ("Cloud", "Planting land", "Meadow", "Forest", "Village and homestay", "Residential area", "Education area",
                   "Factory area", "Structure1(chemical plant)", "Structure2(city wall)", "Metallic ore and Sand pit",
                   "Bare surface", "Traffic", "Water area", "Saline-alkali land", "Sandy land")
    for k in range(num_classes):
        plt.figure(k+3)
        plt.xlabel("Epoch number")
        plt.ylabel("Precison and Recall")
        plt.title(f"the variation of {class_label[k]}")
        plt.plot(epoch_per_eval["epoch"], seq_pre[k], "green", label='precision')
        plt.plot(epoch_per_eval["epoch"], seq_re[k], "red", label='recall')
        plt.legend()
        plt.savefig(os.path.join(path, f"{class_label[k]}_variation.png"))


# def eval_show(epoch_per_eval, path, args):
#     plt.xlabel("epoch number")
#     plt.ylabel("Model accuracy")
#     plt.title("Model accuracy variation chart")
#     plt.plot(epoch_per_eval["epoch"], epoch_per_eval["acc"], "red")
#     plt.savefig(os.path.join(path, "train_epochs_" + str(args.train_epochs) + "_batch_size_" + str(args.batch_size) +
#                               "_lr_type_" + str(args.lr_type) + "_base_lr_" + str(args.base_lr) +
#                               "_lr_decay_rate_" + str(args.lr_decay_rate) + "_weight_decay_" + str(args.weight_decay) + "_Dice.png"))
#     plt.show()



class SaveCallback(Callback):
    def __init__(self, eval_model, ds_eval, args, epochs_per_eval):
        super(SaveCallback, self).__init__()
        self.model = eval_model
        self.ds_eval = ds_eval
        self.acc = 0
        self.train_dir = args.train_dir
        self.epochs_per_eval = epochs_per_eval

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        result = self.model.eval(self.ds_eval)
        self.epochs_per_eval["epoch"].append(cur_epoch)
        self.epochs_per_eval["acc"].append(result["Dice"])
        print(result)

        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)
        if result['Dice'] > self.acc:
            self.acc = result['Dice']
            file_name = os.path.join(self.train_dir, str(cur_epoch) + "__" + str(self.acc) + ".ckpt")
            save_checkpoint(save_obj=cb_params.train_network, ckpt_file_name=file_name)
            print("Save the maximum accuracy checkpoint,the Dice is", self.acc)







