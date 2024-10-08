# import matplotlib.pyplot as plt
import numpy as np
import mindspore
import os
# import imageio
import cv2
import PIL
from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = 20000000000
import mindspore.nn as nn
# ===========================================================================================================
# 评价指标

#计算混淆矩阵（a:prediction,b:label）
def cal_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(np.int32) + b[k], minlength=n ** 2).reshape(n, n)

def OverallAccuracy(confusionMatrix):
    #  返回所有类的整体像素精度OA
    # acc = (TP + TN) / (TP + TN + FP + TN)
    OA = np.diag(confusionMatrix).sum() / confusionMatrix.sum()
    # print(OA)
    return OA

def Kappa(confusion_matrix):
    pe_rows = np.sum(confusion_matrix, axis=0)
    pe_cols = np.sum(confusion_matrix, axis=1)
    sum_total = sum(pe_cols)
    pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
    po = np.trace(confusion_matrix) / float(sum_total)
    return (po - pe) / (1 - pe)

def Mean_Intersection_over_Union(confusion_matrix):
    IoU = np.diag(confusion_matrix) / (
                np.sum(confusion_matrix, axis=1) +
                np.sum(confusion_matrix, axis=0) -
                np.diag(confusion_matrix))
    # print('iou:', IoU)
    # IoU = IoU[1:]
    # print('iou', IoU)
    # MIoU = np.nanmean(IoU)
    MIoU = np.mean(IoU)
    # print('MIOU:', MIoU)
    return IoU, MIoU

def Precision(confusionMatrix):
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis=1)
    # precision = precision[:-1]
    # precision = precision[0:]
    # print(precision)
    return precision

def Recall(confusionMatrix):
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis=0)
    # recall = recall[0:]
    return recall

def F1Score(confusionMatrix):
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis=1)
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis=0)
    f1score = 2 * precision * recall / (precision + recall)
    f1score = f1score[0:]
    return f1score

def Frequency_Weighted_Intersection_over_Union(confusionMatrix):
    freq = np.sum(confusionMatrix, axis=1) / np.sum(confusionMatrix)
    iu = np.diag(confusionMatrix) / (
            np.sum(confusionMatrix, axis=1) +
            np.sum(confusionMatrix, axis=0) -
            np.diag(confusionMatrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU



def Record_result_evaluation(args, hist, target_names, precision, recall, f1ccore, OA, kappa, MIoU, FWIoU):
    with open(args.out_path + '/Record_test_parameters_and_pred_result.txt', 'a') as f:
        f.write('# ========================================================================================== \n')
        f.write(str(args) + '\n')
        f.write('Confusion matrix:\n')
        f.write(str(hist) + '\n')
        f.write('target_names:      ' + str(target_names) + '\n')
        f.write('precision:         ' + str(precision) + '\n')
        f.write('recall:            ' + str(recall) + '\n')
        f.write('f1ccore:           ' + str(f1ccore) + '\n')
        f.write("OA:                " + str(OA) + '\n')
        f.write("kappa:             " + str(kappa) + '\n')
        f.write("MIoU:              " + str(MIoU) + '\n')
        f.write("FWIoU:             " + str(FWIoU) + '\n')


def Record_train_parameters_set(args, local_train_url):
    with open(local_train_url + '/Record_train_parameters_set.txt', 'a') as f:
        f.write('# ========================================================================================== \n')
        f.write('The train parameters seting is show:\n')
        f.write('# Ascend or CPU \n')
        f.write('# dataset \n')

        f.write('train_dir:          ' + str(args.train_dir) + '\n')
        f.write('train_data_file:    ' +str(args.train_data_file) + '\n')
        f.write('val_data_file:      ' +str(args.val_data_file)  + '\n' )
        f.write('eval_per_epoch:     ' + str(args.eval_per_epoch)   + '\n')
        # f.write('save_per_epoch:     ' + str(args.save_per_epoch) + '\n\n')

        f.write('batch_size:         ' + str(args.batch_size) + '\n')
        f.write('crop_size:          ' + str(args.crop_size) + '\n')
        f.write('image_mean:         ' + str(args.image_mean) + '\n')
        f.write('image_std:          ' + str(args.image_std)    + '\n' )
        f.write('min_scale:          ' + str(args.min_scale)   + '\n')
        f.write('max_scale:          ' + str(args.max_scale) + '\n')
        f.write('ignore_label:       ' + str(args.ignore_label) + '\n')
        f.write('num_classes:        ' + str(args.num_classes) + '\n\n')

        f.write('# optimizer \n')
        f.write('train_epochs:       ' + str(args.train_epochs)    + '\n' )
        f.write('lr_type:            ' + str(args.lr_type)   + '\n')
        f.write('base_lr:            ' + str(args.base_lr) + '\n')
        f.write('lr_decay_step:      ' + str(args.lr_decay_step) + '\n')
        f.write('lr_decay_rate:      ' + str(args.lr_decay_rate) + '\n')
        f.write('loss_scale:         ' + str(args.loss_scale)    + '\n' )
        f.write('weight_decay:       ' + str(args.weight_decay)   + '\n')
        # f.write('use_balanced_weights:       ' + str(args.use_balanced_weights) + '\n\n')

        f.write('# model \n')
        f.write('model:              ' + str(args.model) + '\n')
        f.write('freeze_bn:          ' + str(args.freeze_bn)    + '\n' )
        # f.write('pretrainedmodel:    ' +str(args.data_url)+'/'+ str(args.pretrainedmodel_filename)  + '\n')

        f.write('device_target:      ' + str(args.device_target) + '\n')
        f.write('is_distributed:     ' + str(args.is_distributed) + '\n')
        f.write('rank:               ' + str(args.rank) + '\n')
        f.write('group_size:         ' + str(args.group_size)    + '\n' )
        f.write('save_steps:         ' + str(args.save_steps)   + '\n')
        f.write('keep_checkpoint_max:' + str(args.keep_checkpoint_max) + '\n')
        f.write('pretrainedmodel_filename:' + str(args.pretrainedmodel_filename) + '\n')
        # f.write('amp_level:          ' + str(args.amp_level) + '\n\n')

        f.write('# ModelArts \n')
        f.write('modelArts_mode:     ' + str(args.modelArts_mode) + '\n')
        f.write('train_url:          ' + str(args.train_url) + '\n')
        f.write('data_url:           ' + str(args.data_url)  + '\n')
        f.write('train_data_filename:' + str(args.train_data_filename) + '\n')
        f.write('val_data_filename:   ' + str(args.val_data_filename) + '\n')
        # f.write('pretrainedmodel_filename:' + str(args.pretrainedmodel_filename) + '\n\n')



def Record_test_parameters_set(args):
    with open(args.out_path + '/Record_test_parameters_and_pred_result.txt', 'a') as f:
        f.write('# ========================================================================================== \n')
        f.write('MindSpore DeepLabV3+ eval parameters seting is show :\n')
        f.write('# test data \n')
        f.write('data_root:          ' + str(args.data_root) + '\n')
        f.write('data_lst:           ' + str(args.data_lst) + '\n')
        f.write('batch_size:         ' + str(args.batch_size) + '\n')
        f.write('crop_size:          ' + str(args.crop_size) + '\n')
        f.write('image_mean:         ' + str(args.image_mean) + '\n')
        f.write('image_std:          ' + str(args.image_std)    + '\n\n' )

        f.write('scales:             ' + str(args.scales)   + '\n')
        f.write('flip:               ' + str(args.flip) + '\n')
        f.write('ignore_label:       ' + str(args.ignore_label) + '\n')
        f.write('num_classes:        ' + str(args.num_classes) + '\n\n')

        f.write('index:              ' + str(args.index)    + '\n' )
        f.write('out_path:           ' + str(args.train_url)   + '\n')
        f.write('dataset:            ' + str(args.dataset) + '\n')
        f.write('slidesize:          ' + str(args.slidesize) + '\n\n')

        f.write('# model \n')
        f.write('model:              ' + str(args.model) + '\n')
        f.write('freeze_bn:          ' + str(args.freeze_bn)    + '\n' )
        f.write('ckpt_path:          ' + str(args.restore_from)   + '\n\n')


