import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .cross_entropy_loss import binary_cross_entropy


@LOSSES.register_module()
class CombinedBCEDiceBoundaryLoss(nn.Module):
    """Combined Loss: BCEWithLogits + Dice + Boundary Loss.
    
    Total loss = BCE + Dice + Boundary
    Weight ratio: BCE : Dice : Boundary = 1 : 1 : 1
    
    Args:
        bce_weight (float): Weight for BCE loss. Default: 1.0.
        dice_weight (float): Weight for Dice loss. Default: 1.0.
        boundary_weight (float): Weight for boundary pixels in boundary loss. Default: 5.0.
        boundary_kernel (int): Kernel size for boundary extraction. Default: 3.
        boundary_loss_weight (float): Weight for Boundary loss component. Default: 1.0.
        ignore_index (int): Index to ignore. Default: -100.
    """
    
    def __init__(self,
                 bce_weight=1.0,
                 dice_weight=1.0,
                 boundary_weight=5.0,
                 boundary_kernel=3,
                 boundary_loss_weight=1.0,
                 ignore_index=-100):
        super(CombinedBCEDiceBoundaryLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.boundary_kernel = boundary_kernel
        self.boundary_loss_weight = boundary_loss_weight
        self.ignore_index = ignore_index
    
    def get_boundary_mask(self, target):
        """Extract boundary mask from segmentation mask."""
        target_float = target.float()
        
        if target_float.dim() == 3:
            target_float = target_float.unsqueeze(1)  # (N, 1, H, W)
        
        # Create kernel for boundary detection
        kernel = torch.ones(1, 1, self.boundary_kernel, self.boundary_kernel, 
                           device=target.device, dtype=target.dtype)
        kernel = kernel / (self.boundary_kernel * self.boundary_kernel)
        
        # Apply average pooling to get local mean
        local_mean = F.conv2d(target_float, kernel, padding=self.boundary_kernel // 2)
        
        # Boundary is where local mean is between 0 and 1
        boundary = (local_mean > 0.0) & (local_mean < 1.0)
        boundary = boundary | ((target_float > 0.5) & (local_mean < 1.0))
        
        return boundary.squeeze(1)  # (N, H, W)
    
    def dice_loss(self, pred, target, smooth=1.0):
        """Calculate Dice Loss."""
        # Flatten tensors
        if pred.dim() == 4:
            pred = pred.squeeze(1)  # (N, H, W)
        
        # Apply sigmoid
        pred = torch.sigmoid(pred)
        
        # Create mask for valid pixels
        valid_mask = (target != self.ignore_index)
        if valid_mask.sum() == 0:
            return pred.sum() * 0
        
        pred = pred[valid_mask]
        target_float = target[valid_mask].float()
        
        # Calculate Dice coefficient
        intersection = (pred * target_float).sum()
        dice_coeff = (2.0 * intersection + smooth) / (pred.sum() + target_float.sum() + smooth)
        
        return 1.0 - dice_coeff
    
    def boundary_loss(self, pred, target):
        """Calculate Boundary Loss."""
        # Flatten tensors
        if pred.dim() == 4:
            pred = pred.squeeze(1)  # (N, H, W)
        
        # Apply sigmoid
        pred = torch.sigmoid(pred)
        
        # Create mask for valid pixels
        valid_mask = (target != self.ignore_index)
        if valid_mask.sum() == 0:
            return pred.sum() * 0
        
        # Extract boundary mask
        boundary_mask = self.get_boundary_mask(target)
        boundary_mask = boundary_mask & valid_mask
        
        # Calculate BCE loss
        target_float = target.float()
        loss = F.binary_cross_entropy(pred, target_float, reduction='none')
        
        # Apply boundary weight
        weight_map = torch.ones_like(loss)
        weight_map[boundary_mask] = self.boundary_weight
        
        # Weighted loss
        loss = loss * weight_map
        
        # Average over valid pixels
        loss = loss[valid_mask].mean()
        
        return loss
    
    def forward(self, pred, target, weight=None, avg_factor=None, **kwargs):
        """Forward function.
        
        Args:
            pred (torch.Tensor): The prediction with shape (N, 1, H, W).
            target (torch.Tensor): The learning label with shape (N, H, W).
            weight (torch.Tensor, optional): Element-wise weights.
            avg_factor (float, optional): Average factor.
        
        Returns:
            torch.Tensor: The calculated combined loss.
        """
        # Handle target shape: if target is (N, 1, H, W), squeeze to (N, H, W)
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)
        
        # Handle pred shape: ensure pred is (N, 1, H, W) or (N, H, W)
        if pred.dim() == 4:
            pred_flat = pred.squeeze(1)  # (N, H, W) for BCE
        else:
            pred_flat = pred
        
        # Create valid mask
        valid_mask = (target != self.ignore_index)
        if valid_mask.sum() == 0:
            return pred.sum() * 0
        
        # BCE Loss (with logits, so no sigmoid needed in loss function)
        # Use F.binary_cross_entropy_with_logits directly for better control
        target_float = target.float()
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_flat,
            target_float,
            reduction='none'
        )
        
        # Apply weight if provided
        if weight is not None:
            if weight.dim() == 4:
                weight = weight.squeeze(1)
            bce_loss = bce_loss * weight
        
        # Apply valid mask and reduce
        bce_loss = bce_loss[valid_mask]
        if avg_factor is not None:
            bce_loss = bce_loss.sum() / avg_factor
        else:
            bce_loss = bce_loss.mean()
        
        # Dice Loss
        dice_loss_val = self.dice_loss(pred, target)
        
        # Boundary Loss
        boundary_loss_val = self.boundary_loss(pred, target)
        
        # Combined loss with equal weights (1:1:1)
        total_loss = (self.bce_weight * bce_loss + 
                     self.dice_weight * dice_loss_val + 
                     self.boundary_loss_weight * boundary_loss_val)
        
        return total_loss

