import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss


def dice_loss(pred, target, smooth=1.0, ignore_index=-100):
    """Calculate Dice Loss.
    
    Args:
        pred (torch.Tensor): The prediction with shape (N, 1, H, W) or (N, H, W).
        target (torch.Tensor): The learning label with shape (N, H, W).
        smooth (float): Smoothing factor to avoid division by zero. Default: 1.0.
        ignore_index (int): Index to ignore. Default: -100.
    
    Returns:
        torch.Tensor: The calculated Dice loss.
    """
    # Flatten tensors
    if pred.dim() == 4:
        pred = pred.squeeze(1)  # (N, H, W)
    
    # Apply sigmoid if not already applied
    pred = torch.sigmoid(pred)
    
    # Create mask for valid pixels
    valid_mask = (target != ignore_index)
    if valid_mask.sum() == 0:
        return pred.sum() * 0  # Return zero loss if no valid pixels
    
    pred = pred[valid_mask]
    target = target[valid_mask].float()
    
    # Calculate Dice coefficient
    intersection = (pred * target).sum()
    dice_coeff = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    # Dice loss = 1 - Dice coefficient
    loss = 1.0 - dice_coeff
    
    return loss


@LOSSES.register_module()
class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation.
    
    Args:
        smooth (float): Smoothing factor. Default: 1.0.
        loss_weight (float): Weight of the loss. Default: 1.0.
        ignore_index (int): Index to ignore. Default: -100.
    """
    
    def __init__(self, smooth=1.0, loss_weight=1.0, ignore_index=-100):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
    
    def forward(self, pred, target, weight=None, avg_factor=None, **kwargs):
        """Forward function.
        
        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label.
            weight (torch.Tensor, optional): Element-wise weights.
            avg_factor (float, optional): Average factor.
        
        Returns:
            torch.Tensor: The calculated loss.
        """
        loss = dice_loss(pred, target, smooth=self.smooth, ignore_index=self.ignore_index)
        loss = self.loss_weight * loss
        return loss

