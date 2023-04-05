import torch
import numpy as np
import difflib
import numpy as np
import os
import SimpleITK as sitk
import scipy.spatial


# def dice_coef(y_pred, y_true):

#     # Convert to Binary
#     zeros = torch.zeros(y_pred.size())
#     ones = torch.ones(y_pred.size())

#     y_pred = y_pred.cpu()
#     y_pred = torch.where(y_pred > 0.5, ones, zeros)

#     if torch.cuda.is_available():
#         y_pred = y_pred.cuda()

#     y_true = y_true.cpu()
#     y_true = torch.where(y_true > 0, ones, zeros)
    
#     if torch.cuda.is_available():
#         y_true = y_true.cuda()


#     # Calculate Dice Coefficient Score
#     smooth = 1.
    
#     #y_pred = y_pred.view(-1)
#     #y_true = y_true.view(-1)
#     #print(y_pred.size())

#     intersection = (y_pred * y_true).sum()

#     return (2 * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)


def getHausdorff(testImage, resultImage):
    """Compute the 95% Hausdorff distance."""
    hd = dict()

    k = 1

    lTestImage = testImage
    lResultImage = resultImage

    # Hausdorff distance is only defined when something is detected
    statistics = sitk.StatisticsImageFilter()
    statistics.Execute(lTestImage)
    lTestSum = statistics.GetSum()
    statistics.Execute(lResultImage)
    lResultSum = statistics.GetSum()

    # Edge detection is done by ORIGINAL - ERODED, keeping the outer boundaries of lesions. Erosion is performed in 2D
    eTestImage = sitk.BinaryErode(lTestImage, (1, 1, 0))
    eResultImage = sitk.BinaryErode(lResultImage, (1, 1, 0))

    hTestImage = sitk.Subtract(lTestImage, eTestImage)
    hResultImage = sitk.Subtract(lResultImage, eResultImage)

    hTestArray = sitk.GetArrayFromImage(hTestImage)
    hResultArray = sitk.GetArrayFromImage(hResultImage)

    # Convert voxel location to world coordinates. Use the coordinate system of the test image
    # np.nonzero   = elements of the boundary in numpy order (zyx)
    # np.flipud    = elements in xyz order
    # np.transpose = create tuples (x,y,z)
    # testImage.TransformIndexToPhysicalPoint converts (xyz) to world coordinates (in mm)
    # (Simple)ITK does not accept all Numpy arrays; therefore we need to convert the coordinate tuples into a Python list before passing them to TransformIndexToPhysicalPoint().
    testCoordinates = [testImage.TransformIndexToPhysicalPoint(x.tolist()) for x in
                       np.transpose(np.flipud(np.nonzero(hTestArray)))]
    resultCoordinates = [testImage.TransformIndexToPhysicalPoint(x.tolist()) for x in
                         np.transpose(np.flipud(np.nonzero(hResultArray)))]

    # Use a kd-tree for fast spatial search
    def getDistancesFromAtoB(a, b):
        kdTree = scipy.spatial.KDTree(a, leafsize=100)
        return kdTree.query(b, k=1, eps=0, p=2)[0]

    # Compute distances from test to result and vice versa.
    dTestToResult = getDistancesFromAtoB(testCoordinates, resultCoordinates)
    dResultToTest = getDistancesFromAtoB(resultCoordinates, testCoordinates)
    hd[k] = max(np.percentile(dTestToResult, 95), np.percentile(dResultToTest, 95))

    return hd


def bin_total(y_true, y_prob, n_bins):
    bins = np.linspace(0., 1. + 1e-8, n_bins + 1)

    # In sklearn.calibration.calibration_curve,
    # the last value in the array is always 0.
    binids = np.digitize(y_prob, bins) - 1

    return np.bincount(binids, minlength=len(bins))


import torch
import numpy as np
import os

def dice_coef(y_pred, y_true):

    # Convert to Binary
    zeros = torch.zeros(y_pred.size())
    ones = torch.ones(y_pred.size())

    y_pred = y_pred.cpu()
    y_pred = torch.where(y_pred > 0.5, ones, zeros)

    if torch.cuda.is_available():
        y_pred = y_pred.cuda()

    y_true = y_true.cpu()
    y_true = torch.where(y_true > 0, ones, zeros)
    
    if torch.cuda.is_available():
        y_true = y_true.cuda()


    # Calculate Dice Coefficient Score
    smooth = 1.
    
    score = 0
    for i in range(y_pred.shape[0]):
        pred = y_pred[i]
        true = y_true[i]

        intersection = (pred * true).sum()
        score += (2 * intersection + smooth) / (pred.sum() + true.sum() + smooth)

    return score/(y_pred.shape[0])

def TP(y_true, y_pred):
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    true_positives = torch.sum((torch.round(torch.clamp(y_true_f * y_pred_f, 0, 1))))
    return true_positives


def FP(y_true, y_pred):
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    y_pred_f01 = torch.round(torch.clamp(y_pred_f, 0, 1))
    tp_f01 = torch.round(torch.clamp(y_true_f * y_pred_f, 0, 1))
    false_positives = torch.sum((torch.round(torch.clamp(y_pred_f01 - tp_f01, 0, 1))))
    return false_positives


def TN(y_true, y_pred):
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    y_pred_f01 = torch.round(torch.clamp(y_pred_f, 0, 1))
    all_one = torch.ones_like(y_pred_f01)
    y_pred_f_1 = -1 * (y_pred_f01 - all_one)
    y_true_f_1 = -1 * (y_true_f - all_one)
    true_negatives = torch.sum(torch.round(torch.clamp(y_true_f_1 + y_pred_f_1, 0, 1)))
    return true_negatives


def FN(y_true, y_pred):
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
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