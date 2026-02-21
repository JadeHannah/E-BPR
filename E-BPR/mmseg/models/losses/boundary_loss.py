import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss


def get_boundary_mask(target, kernel_size=3):
    """Extract boundary mask from segmentation mask.
    
    Args:
        target (torch.Tensor): Ground truth mask with shape (N, H, W).
        kernel_size (int): Kernel size for boundary extraction. Default: 3.
    
    Returns:
        torch.Tensor: Boundary mask with shape (N, H, W).
    """
    # Convert to float
    target = target.float()
    
    # Create kernel for boundary detection
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=target.device, dtype=target.dtype)
    kernel = kernel / (kernel_size * kernel_size)
    
    # Expand target to (N, 1, H, W) for convolution
    if target.dim() == 3:
        target = target.unsqueeze(1)  # (N, 1, H, W)
    
    # Apply average pooling to get local mean
    local_mean = F.conv2d(target, kernel, padding=kernel_size // 2)
    
    # Boundary is where local mean is between 0 and 1 (i.e., not all 0 or all 1)
    boundary = (local_mean > 0.0) & (local_mean < 1.0)
    
    # Also include pixels that are 1 but have neighbors that are 0
    boundary = boundary | ((target > 0.5) & (local_mean < 1.0))
    
    return boundary.squeeze(1)  # (N, H, W)


def boundary_loss(pred, target, boundary_weight=5.0, kernel_size=3, ignore_index=-100):
    """Calculate Boundary Loss.
    
    Args:
        pred (torch.Tensor): The prediction with shape (N, 1, H, W) or (N, H, W).
        target (torch.Tensor): The learning label with shape (N, H, W).
        boundary_weight (float): Weight for boundary pixels. Default: 5.0.
        kernel_size (int): Kernel size for boundary extraction. Default: 3.
        ignore_index (int): Index to ignore. Default: -100.
    
    Returns:
        torch.Tensor: The calculated Boundary loss.
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
    
    # Extract boundary mask
    boundary_mask = get_boundary_mask(target, kernel_size=kernel_size)
    
    # Apply valid mask
    boundary_mask = boundary_mask & valid_mask
    
    if boundary_mask.sum() == 0:
        # If no boundary pixels, return regular BCE loss
        target_float = target.float()
        loss = F.binary_cross_entropy(pred, target_float, reduction='none')
        loss = loss[valid_mask].mean()
        return loss
    
    # Calculate BCE loss
    target_float = target.float()
    loss = F.binary_cross_entropy(pred, target_float, reduction='none')
    
    # Apply boundary weight
    weight_map = torch.ones_like(loss)
    weight_map[boundary_mask] = boundary_weight
    
    # Weighted loss
    loss = loss * weight_map
    
    # Average over valid pixels
    loss = loss[valid_mask].mean()
    
    return loss


@LOSSES.register_module()
class BoundaryLoss(nn.Module):
    """Boundary Loss for binary segmentation.
    
    Args:
        boundary_weight (float): Weight for boundary pixels. Default: 5.0.
        kernel_size (int): Kernel size for boundary extraction. Default: 3.
        loss_weight (float): Weight of the loss. Default: 1.0.
        ignore_index (int): Index to ignore. Default: -100.
    """
    
    def __init__(self, boundary_weight=5.0, kernel_size=3, loss_weight=1.0, ignore_index=-100):
        super(BoundaryLoss, self).__init__()
        self.boundary_weight = boundary_weight
        self.kernel_size = kernel_size
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
        loss = boundary_loss(
            pred, 
            target, 
            boundary_weight=self.boundary_weight,
            kernel_size=self.kernel_size,
            ignore_index=self.ignore_index
        )
        loss = self.loss_weight * loss
        return loss

