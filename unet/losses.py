"""
Script for other loss functions to use
"""

import torch

def tversky_loss(y_pred, y_true, alpha=0.7, beta=0.3, smooth=1e-6):
    y_pred = torch.sigmoid(y_pred)
    y_true = y_true.float()
    tp = (y_pred * y_true).sum()
    fp = ((1 - y_true) * y_pred).sum()
    fn = (y_true * (1 - y_pred)).sum()
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1 - tversky