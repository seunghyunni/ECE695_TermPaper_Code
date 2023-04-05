import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def dice_loss(y_pred, y_true):
    # Calculate Dice Coefficient Score
    smooth = 1.
    
    #y_pred = y_pred.view(-1)
    #y_true = y_true.view(-1)
    #print(y_pred.size())

    intersection = (y_pred * y_true).sum()
    
    return 1 - (2 * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)