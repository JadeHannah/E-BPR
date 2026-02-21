import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer
from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead


class MultiScalePyramidPooling(nn.Module):
    """Multi-Scale Pyramid Pooling module with dilated convolutions.
    
    Applies dilated convolutions at different rates to extract multi-scale
    context information without spatial downsampling.
    
    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        dilations (tuple): Dilation rates. Default: (1, 6, 12, 18).
        norm_cfg (dict): Config of norm layers.
    """
    def __init__(self, in_channels, out_channels, dilations=(1, 6, 12, 18), norm_cfg=None):
        super(MultiScalePyramidPooling, self).__init__()
        self.dilations = dilations
        self.aspp_convs = nn.ModuleList()
        
        # Dilated convolutions at different rates
        for dilation in dilations:
            self.aspp_convs.append(
                ConvModule(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='ReLU')
                )
            )
        
        # Global average pooling branch for global context
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                in_channels,
                out_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=dict(type='ReLU')
            )
        )
        
        # Dense connection fusion: concatenate all multi-scale features
        self.fusion_conv = ConvModule(
            out_channels * (len(dilations) + 1),
            out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU')
        )

    def forward(self, x):
        """Forward function with dense connection."""
        aspp_outs = []
        
        # Multi-scale dilated convolutions
        for aspp_conv in self.aspp_convs:
            aspp_outs.append(aspp_conv(x))
        
        # Global average pooling for global context
        global_feat = self.global_pool(x)
        global_feat = resize(
            global_feat,
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        aspp_outs.append(global_feat)
        
        # Dense connection: concatenate all multi-scale features
        out = torch.cat(aspp_outs, dim=1)
        out = self.fusion_conv(out)
        return out


@HEADS.register_module()
class MSPPHead(BaseDecodeHead):
    """Multi-Scale Pyramid Pooling Head for HRNet.
    
    This head implements:
    1. Multi-scale dilated convolutions (MSPP) on low-resolution branches
       to extract context without spatial downsampling
    2. Dense connection fusion of multi-scale context information
    3. Bilinear upsampling to align features to highest resolution (1/4)
    4. Final fusion of all aligned multi-scale features
    
    Args:
        low_res_branches (list): Indices of low-resolution branches to process.
            Default: [1, 2, 3] (1/8, 1/16, 1/32).
        dilations (tuple): Dilation rates for MSPP module. Default: (1, 6, 12, 18).
        target_channels (int): Target channels for alignment (highest resolution branch channels).
            Default: 48.
    """
    
    def __init__(self, 
                 low_res_branches=[1, 2, 3],
                 dilations=(1, 6, 12, 18),
                 target_channels=48,
                 input_transform='multiple_select',
                 **kwargs):
        # Ensure input_transform is set to 'multiple_select' for multi-branch input
        kwargs['input_transform'] = input_transform
        super(MSPPHead, self).__init__(**kwargs)
        self.low_res_branches = low_res_branches
        self.dilations = dilations
        self.target_channels = target_channels
        
        # MSPP modules for low-resolution branches (1/8, 1/16, 1/32)
        self.mspp_modules = nn.ModuleList()
        for branch_idx in self.low_res_branches:
            if branch_idx < len(self.in_channels):
                in_ch = self.in_channels[branch_idx]
                self.mspp_modules.append(
                    MultiScalePyramidPooling(
                        in_ch,
                        self.target_channels,
                        dilations=self.dilations,
                        norm_cfg=self.norm_cfg
                    )
                )
            else:
                self.mspp_modules.append(None)
        
        # Channel alignment for highest resolution branch (1/4)
        if len(self.in_channels) > 0:
            high_res_ch = self.in_channels[0]
            if high_res_ch != self.target_channels:
                self.high_res_align = ConvModule(
                    high_res_ch,
                    self.target_channels,
                    kernel_size=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=dict(type='ReLU')
                )
            else:
                self.high_res_align = None
        else:
            self.high_res_align = None
        
        # Final fusion module: concatenate all aligned features
        num_fused_branches = 1 + len([b for b in self.low_res_branches if b < len(self.in_channels)])
        self.fusion_conv = ConvModule(
            self.target_channels * num_fused_branches,
            self.channels,
            kernel_size=3,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type='ReLU')
        )

    def forward(self, inputs):
        """Forward function.
        
        Args:
            inputs (list[Tensor]): List of multi-level features from HRNet.
                - inputs[0]: 1/4 resolution (highest)
                - inputs[1]: 1/8 resolution
                - inputs[2]: 1/16 resolution
                - inputs[3]: 1/32 resolution (lowest)
        
        Returns:
            Tensor: Segmentation logits.
        """
        # Transform inputs using BaseDecodeHead's mechanism
        inputs = self._transform_inputs(inputs)
        
        # Process highest resolution branch (1/4)
        high_res_feat = inputs[0]
        if self.high_res_align is not None:
            high_res_feat = self.high_res_align(high_res_feat)
        
        # Get target spatial size (highest resolution)
        target_size = high_res_feat.shape[2:]
        
        # Process low-resolution branches with MSPP and bilinear upsampling
        processed_features = [high_res_feat]
        
        for i, branch_idx in enumerate(self.low_res_branches):
            if branch_idx >= len(inputs) or self.mspp_modules[i] is None:
                continue
            
            # Get low-resolution feature
            low_res_feat = inputs[branch_idx]
            
            # Apply MSPP module: multi-scale dilated convolutions with dense connection
            mspp_feat = self.mspp_modules[i](low_res_feat)
            
            # Bilinear upsampling to align to highest resolution
            mspp_feat = resize(
                mspp_feat,
                size=target_size,
                mode='bilinear',
                align_corners=self.align_corners
            )
            
            processed_features.append(mspp_feat)
        
        # Dense connection: concatenate all processed features
        fused_feat = torch.cat(processed_features, dim=1)
        
        # Final fusion
        output = self.fusion_conv(fused_feat)
        
        # Classification
        output = self.cls_seg(output)
        return output

