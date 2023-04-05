import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def TP(y_true, y_pred):
    y_true_f = y_true.reshape(-1)
    y_pred_f = y_pred.reshape(-1)
    true_positives = torch.sum((torch.round(torch.clamp(y_true_f * y_pred_f, 0, 1))))
    return true_positives


def FP(y_true, y_pred):
    y_true_f = y_true.reshape(-1)
    y_pred_f = y_pred.reshape(-1)
    y_pred_f01 = torch.round(torch.clamp(y_pred_f, 0, 1))
    tp_f01 = torch.round(torch.clamp(y_true_f * y_pred_f, 0, 1))
    false_positives = torch.sum((torch.round(torch.clamp(y_pred_f01 - tp_f01, 0, 1))))
    return false_positives


def TN(y_true, y_pred):
    y_true_f = y_true.reshape(-1)
    y_pred_f = y_pred.reshape(-1)
    y_pred_f01 = torch.round(torch.clamp(y_pred_f, 0, 1))
    all_one = torch.ones_like(y_pred_f01)
    y_pred_f_1 = -1 * (y_pred_f01 - all_one)
    y_true_f_1 = -1 * (y_true_f - all_one)
    true_negatives = torch.sum(torch.round(torch.clamp(y_true_f_1 + y_pred_f_1, 0, 1)))
    return true_negatives


def FN(y_true, y_pred):
    y_true_f = y_true.reshape(-1)
    y_pred_f = y_pred.reshape(-1)
    # y_pred_f01 = keras.round(keras.clip(y_pred_f, 0, 1))
    tp_f01 = torch.round(torch.clamp(y_true_f * y_pred_f, 0, 1))
    false_negatives = torch.sum(torch.round(torch.clamp(y_true_f - tp_f01, 0, 1)))
    return false_negatives


def sensitivity(y_true, y_pred):
    tp = TP(y_true, y_pred)
    fn = FN(y_true, y_pred)
    if tp + fn ==0: 
        res = 0 
    else: 
        res = tp / (tp + fn)
    return res


def ppv(y_true, y_pred):
    tp = TP(y_true, y_pred)
    fp = FP(y_true, y_pred)
    return tp / (tp + fp)


def specificity(y_true, y_pred):
    tn = TN(y_true, y_pred)
    fp = FP(y_true, y_pred)
    return tn / (tn + fp)


def dice_loss(y_pred, y_true):
    # Calculate Dice Coefficient Score
    smooth = 1.
    
    #y_pred = y_pred.view(-1)
    #y_true = y_true.view(-1)
    #print(y_pred.size())

    intersection = (y_pred * y_true).sum()
    
    return 1 - (2 * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)


def FP_dice(y_pred, y_true):
    # Convert to Binary
    zeros = torch.zeros(y_pred.size())
    ones = torch.ones(y_pred.size())

    y_pred = y_pred.cpu()
    # y_pred = torch.where(y_pred > 0.5, ones, zeros)

    if torch.cuda.is_available():
        y_pred = y_pred.cuda()

    y_true = y_true.cpu()
    y_true = torch.where(y_true > 0, zeros, ones)

    if torch.cuda.is_available():
        y_true = y_true.cuda()

    # Calculate Dice Coefficient Score
    smooth = 1.

    # y_pred = y_pred.view(-1)
    # y_true = y_true.view(-1)
    # print(y_pred.size())

    intersection = (y_pred * y_true).sum()

    return (2 * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)


def FN_dice(y_pred, y_true):
    # Convert to Binary
    zeros = torch.zeros(y_pred.size())
    ones = torch.ones(y_pred.size())

    y_pred = y_pred.cpu()
    # y_pred = torch.where(y_pred > 0.5, ones, zeros)

    if torch.cuda.is_available():
        y_pred = y_pred.cuda()

    y_true = y_true.cpu()
    y_true = torch.where(y_true > 0, ones, zeros)

    if torch.cuda.is_available():
        y_true = y_true.cuda()

    # Calculate Dice Coefficient Score
    smooth = 1.

    # y_pred = y_pred.view(-1)
    # y_true = y_true.view(-1)
    # print(y_pred.size())

    intersection = (y_pred * y_true).sum()

    return (2 * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)
