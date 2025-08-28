"""
Script for other loss functions to use
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCE_FP(nn.Module):
    def __init__(self, false_positive_weight=2.0, false_negative_weight=1.0, eps=1e-7):
        """
        Custom BCE loss that penalizes false positives more heavily.

        false_positive_weight: penalty multiplier for false positives (label=0, pred=1)
        false_negative_weight: penalty multiplier for false negatives (label=1, pred=0)
        eps: small value to prevent division by zero
        """
        super().__init__()
        self.fp_weight = false_positive_weight
        self.fn_weight = false_negative_weight
        self.eps = eps

    def forward(self, logits, targets):
        # logits: shape [batch_size]
        # targets: binary labels [batch_size]
        probs = torch.sigmoid(logits)
        targets = targets.float()

        # Apply asymmetric weighting
        loss_pos = -self.fn_weight * targets * torch.log(probs + self.eps)
        loss_neg = -self.fp_weight * (1 - targets) * torch.log(1 - probs + self.eps)

        loss = loss_pos + loss_neg
        return loss.mean()


def tversky_loss(y_pred, y_true, alpha=0.7, beta=0.3, smooth=1e-6):
    y_pred = torch.sigmoid(y_pred)
    y_true = y_true.float()
    tp = (y_pred * y_true).sum()
    fp = ((1 - y_true) * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1 - tversky

def tversky_loss_penalize_fn(y_pred, y_true, alpha=0.2, beta=0.8, smooth=1e-6):
    y_pred = torch.sigmoid(y_pred)
    y_true = y_true.float()
    tp = (y_pred * y_true).sum()
    fp = ((1 - y_true) * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1 - tversky


def dice_loss(pred, target, smooth=1):
    """
    Computes the Dice Loss for binary segmentation.
    Args:
        pred: Tensor of predictions (batch_size, 1, H, W).
        target: Tensor of ground truth (batch_size, 1, H, W).
        smooth: Smoothing factor to avoid division by zero.
    Returns:
        Scalar Dice Loss.
    """
    # Apply sigmoid to convert logits to probabilities
    pred = torch.sigmoid(pred)

    # flatten label and prediction tensors
    inputs = pred.view(-1)
    targets = target.view(-1)

    intersection = (inputs * targets).sum()
    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

    return 1 - dice

def logcosh_dice_loss(pred, target, smooth=1):
    """
    Wrap Dice loss in logcosh
    """
    dice = dice_loss(pred, target)

    return  torch.log(torch.cosh(1-dice))

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

class DiceWeightedBCE03Loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceWeightedBCE03Loss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        BCE_weighted = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.3], device=device))

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = BCE_weighted(inputs, targets)
        Dice_WeightedBCE = BCE + dice_loss

        return Dice_WeightedBCE
