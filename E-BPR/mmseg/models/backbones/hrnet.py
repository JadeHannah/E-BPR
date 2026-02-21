import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
                      kaiming_init)
from mmcv.runner import load_checkpoint
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmseg.ops import Upsample, resize
from mmseg.utils import get_root_logger
from ..builder import BACKBONES
from .resnet import BasicBlock, Bottleneck

# 内容感知上采样模块（Content-Aware Upsampling）
class ContentAwareUpsampling(nn.Module):
    """Content-Aware Upsampling module with learnable adaptive kernels.
    
    The upsampling kernel is learned from input features, allowing each output
    pixel to be adaptively reconstructed based on large receptive field context.
    Improved version with more efficient implementation.
    """
    def __init__(self, in_channels, out_channels, kernel_size=5, up_factor=2):
        super(ContentAwareUpsampling, self).__init__()
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Compress channels for kernel prediction
        self.compress = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        
        # Kernel prediction with large receptive field (dilated conv for context)
        self.kernel_encoder = nn.Sequential(
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, 
                     padding=2, dilation=2, bias=False),  # Large receptive field
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, up_factor ** 2 * kernel_size ** 2,
                     kernel_size=3, padding=1, bias=False)
        )
        
        # Output projection
        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        Args:
            x: Input feature map (N, C, H, W)
        Returns:
            out: Upsampled feature map (N, C_out, H*up_factor, W*up_factor)
        """
        N, C, H, W = x.size()
        out_h, out_w = H * self.up_factor, W * self.up_factor
        
        # Compress channels
        compressed = self.compress(x)
        compressed = self.norm1(compressed)
        compressed = F.relu(compressed, inplace=True)
        
        # Predict adaptive kernels: (N, up^2 * k^2, H, W)
        kernel = self.kernel_encoder(compressed)
        
        # Reshape and normalize: (N, up^2, k^2, H, W)
        kernel = kernel.view(N, self.up_factor ** 2, self.kernel_size ** 2, H, W)
        kernel = F.softmax(kernel, dim=2)  # Normalize over kernel positions
        
        # Upsample input using nearest neighbor
        x_up = F.interpolate(x, size=(out_h, out_w), mode='nearest')
        
        # Unfold to get patches: (N, C*k*k, out_h*out_w)
        x_unfold = F.unfold(
            x_up, 
            kernel_size=self.kernel_size, 
            padding=self.kernel_size // 2
        )  # (N, C*k*k, out_h*out_w)
        
        x_unfold = x_unfold.view(N, C, self.kernel_size ** 2, out_h, out_w)
        
        # Upsample kernel to output size: (N, up^2, k^2, out_h, out_w)
        kernel_up = F.interpolate(
            kernel.view(N, self.up_factor ** 2 * self.kernel_size ** 2, H, W),
            size=(out_h, out_w),
            mode='bilinear',
            align_corners=False
        )
        kernel_up = kernel_up.view(N, self.up_factor ** 2, self.kernel_size ** 2, out_h, out_w)
        
        # Apply adaptive kernels: for each upsampled position
        # Reshape for broadcasting: (N, 1, C, k^2, out_h, out_w) * (N, up^2, 1, k^2, out_h, out_w)
        x_unfold_expanded = x_unfold.unsqueeze(1)  # (N, 1, C, k^2, out_h, out_w)
        kernel_expanded = kernel_up.unsqueeze(2)  # (N, up^2, 1, k^2, out_h, out_w)
        
        # Weighted sum over kernel positions: (N, up^2, C, out_h, out_w)
        out = (x_unfold_expanded * kernel_expanded).sum(dim=3)
        
        # Reshape: (N, C*up^2, out_h, out_w)
        out = out.view(N, C * self.up_factor ** 2, out_h, out_w)
        
        # Pixel shuffle to get (N, C, out_h, out_w)
        out = F.pixel_shuffle(out, self.up_factor)
        
        # Project to output channels
        out = self.project(out)
        out = self.norm2(out)
        out = self.relu(out)
        
        return out

# 改进的深度可分卷积（用于下采样）
class ImprovedDepthwiseSeparableConv(nn.Module):
    """Improved Depthwise Separable Convolution for downsampling.
    
    Each input channel is processed separately with stride=2, 3x3 convolution,
    then channel number is adjusted with 1x1 pointwise convolution.
    This maintains channel independence and reduces parameters.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(ImprovedDepthwiseSeparableConv, self).__init__()
        # Depthwise convolution: each channel separately with stride=2
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding,
            groups=in_channels, bias=False
        )
        self.dw_norm = nn.BatchNorm2d(in_channels)
        self.dw_relu = nn.ReLU(inplace=True)
        
        # Pointwise convolution: adjust channel number
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pw_norm = nn.BatchNorm2d(out_channels)
        self.pw_relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.dw_norm(x)
        x = self.dw_relu(x)
        x = self.pointwise(x)
        x = self.pw_norm(x)
        x = self.pw_relu(x)
        return x

# 跨分支通道注意力模块（Cross-Branch Channel Attention）
def _channel_shuffle(x, groups=2):
    """Channel shuffle utility used in Conditional Channel Weighting."""
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class ConditionalChannelWeighting(nn.Module):
    """Conditional Channel Weighting (CCW) from Lite-HRNet.

    对每个分支的通道做 split，部分通道保持原样，部分通道通过
    跨分支的条件权重进行重标定，然后通过 channel shuffle 重新混合。
    """

    def __init__(self, channels_list, reduction=4):
        super(ConditionalChannelWeighting, self).__init__()
        self.num_branches = len(channels_list)
        self.channels_list = channels_list
        self.reduction = reduction

        # 对每个分支按通道一分为二
        self.split_channels = [c // 2 for c in channels_list]
        self.remaining_channels = [c - s for c, s in zip(channels_list, self.split_channels)]

        total_remaining = sum(self.remaining_channels)

        # 共享的全局上下文编码（所有分支的“被重标定”通道拼接后一起做）
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        mid_channels = max(total_remaining // reduction, 1)
        self.shared_fc1 = nn.Conv2d(total_remaining, mid_channels, kernel_size=1, bias=False)
        self.shared_bn1 = nn.BatchNorm2d(mid_channels)
        self.shared_relu = nn.ReLU(inplace=True)

        # 为每个分支生成对应的权重
        self.branch_fcs = nn.ModuleList()
        for c_remain in self.remaining_channels:
            self.branch_fcs.append(
                nn.Sequential(
                    nn.Conv2d(mid_channels, c_remain, kernel_size=1, bias=False),
                    nn.BatchNorm2d(c_remain),
                    nn.Sigmoid()
                )
            )

    def forward(self, branch_features):
        # split 通道
        xs_1 = []
        xs_2 = []
        for feat, c_split in zip(branch_features, self.split_channels):
            x1, x2 = torch.split(feat, [c_split, feat.size(1) - c_split], dim=1)
            xs_1.append(x1)
            xs_2.append(x2)

        # 所有分支的 x2 做全局池化并拼接
        pooled_list = [self.global_pool(x2) for x2 in xs_2]
        pooled = torch.cat(pooled_list, dim=1)

        # 共享 MLP
        z = self.shared_fc1(pooled)
        z = self.shared_bn1(z)
        z = self.shared_relu(z)

        # 为每个分支生成条件权重，并重标定 x2
        out_branches = []
        for x1, x2, fc in zip(xs_1, xs_2, self.branch_fcs):
            w = fc(z)
            w = w.expand_as(x2)
            x2_weighted = x2 * w
            out = torch.cat([x1, x2_weighted], dim=1)
            out = _channel_shuffle(out, groups=2)
            out_branches.append(out)

        return out_branches


# SE注意力（保留用于兼容性）
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class HRModule(nn.Module):
    """High-Resolution Module for HRNet.

    In this module, every branch has 4 BasicBlocks/Bottlenecks. Fusion/Exchange
    is in this module.
    """

    def __init__(self,
                 num_branches,
                 blocks,
                 num_blocks,
                 in_channels,
                 num_channels,
                 multiscale_output=True,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(HRModule, self).__init__()
        self._check_branches(num_branches, num_blocks, in_channels,
                             num_channels)

        self.in_channels = in_channels
        self.num_branches = num_branches

        self.multiscale_output = multiscale_output
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.with_cp = with_cp
        self.branches = self._make_branches(num_branches, blocks, num_blocks,
                                            num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=False)

        # Conditional Channel Weighting (Lite-HRNet style), replaces previous
        # cross-branch channel attention + spatial attention.
        self.ccw = ConditionalChannelWeighting(
            [in_channels[i] for i in range(num_branches)]
        )

    def _check_branches(self, num_branches, num_blocks, in_channels,
                        num_channels):
        """Check branches configuration."""
        if num_branches != len(num_blocks):
            error_msg = f'NUM_BRANCHES({num_branches}) <> NUM_BLOCKS(' \
                        f'{len(num_blocks)})'
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) <> NUM_CHANNELS(' \
                        f'{len(num_channels)})'
            raise ValueError(error_msg)

        if num_branches != len(in_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) <> NUM_INCHANNELS(' \
                        f'{len(in_channels)})'
            raise ValueError(error_msg)

    def _make_one_branch(self,
                         branch_index,
                         block,
                         num_blocks,
                         num_channels,
                         stride=1):
        """Build one branch."""
        downsample = None
        if stride != 1 or \
                self.in_channels[branch_index] != \
                num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    self.in_channels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(self.norm_cfg, num_channels[branch_index] *
                                 block.expansion)[1])

        layers = []
        layers.append(
            block(
                self.in_channels[branch_index],
                num_channels[branch_index],
                stride,
                downsample=downsample,
                with_cp=self.with_cp,
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg))
        self.in_channels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.in_channels[branch_index],
                    num_channels[branch_index],
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        """Build multiple branch."""
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        """Build fuse layer with Content-Aware Upsampling (upsampling)
        and original HRNet-style downsampling (3x3 stride-2 conv).
        """
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []
        num_out_branches = num_branches if self.multiscale_output else 1
        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    # Upsampling path: use Content-Aware Upsampling
                    up_factor = 2 ** (j - i)
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(in_channels[j], in_channels[i],
                                      kernel_size=1, stride=1, padding=0, bias=False),
                            nn.BatchNorm2d(in_channels[i]),
                            nn.ReLU(inplace=True),
                            ContentAwareUpsampling(
                                in_channels[i], in_channels[i],
                                kernel_size=5, up_factor=up_factor
                            )
                        )
                    )
                elif j == i:
                    # Same resolution: no transformation needed
                    fuse_layer.append(None)
                else:
                    # Downsampling path: use original 3x3 stride-2 conv stacks
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            out_ch = in_channels[i]
                            conv_downsamples.append(
                                nn.Sequential(
                                    nn.Conv2d(in_channels[j], out_ch,
                                              kernel_size=3, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(out_ch)
                                )
                            )
                        else:
                            out_ch = in_channels[j]
                            conv_downsamples.append(
                                nn.Sequential(
                                    nn.Conv2d(in_channels[j], out_ch,
                                              kernel_size=3, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(out_ch),
                                    nn.ReLU(inplace=True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        """Forward function with enhanced fusion."""
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        # Process each branch
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        # Fuse features: same resolution branches use element-wise addition
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            fused_features = []
            target_size = None
            for j in range(self.num_branches):
                if i == j:
                    # Same resolution: direct use
                    feat = x[j]
                    if target_size is None:
                        target_size = feat.shape[2:]  # Get target spatial size
                elif j > i:
                    # Upsampling: use content-aware upsampling
                    feat = self.fuse_layers[i][j](x[j])
                    # Ensure exact size match
                    if target_size is None:
                        target_size = feat.shape[2:]
                    else:
                        feat = F.interpolate(
                            feat, size=target_size, mode='bilinear', 
                            align_corners=False
                        )
                else:
                    # Downsampling: use improved depthwise separable conv
                    feat = self.fuse_layers[i][j](x[j])
                    # Ensure exact size match
                    if target_size is None:
                        target_size = feat.shape[2:]
                    else:
                        feat = F.interpolate(
                            feat, size=target_size, mode='bilinear', 
                            align_corners=False
                        )
                fused_features.append(feat)
            
            # Element-wise addition for same resolution branches
            y = fused_features[0]
            for feat in fused_features[1:]:
                # Double check size match before addition
                if y.shape[2:] != feat.shape[2:]:
                    feat = F.interpolate(
                        feat, size=y.shape[2:], mode='bilinear', 
                        align_corners=False
                    )
                y = y + feat
            y = self.relu(y)
            
            x_fuse.append(y)

        # Conditional Channel Weighting across branches (Lite-HRNet style)
        x_fuse = self.ccw(x_fuse)

        return x_fuse


# 添加DUC模块
class DUC(nn.Module):
    def __init__(self, inplanes, planes, upscale_factor=2):
        super(DUC, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(planes)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x


# 添加MixedDilatedBlock模块
class MixedDilatedBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(MixedDilatedBlock, self).__init__()
        self.conv_1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, dilation=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(planes)
        self.conv_2 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=2, dilation=2, bias=False)
        self.bn_2 = nn.BatchNorm2d(planes)
        self.conv_3 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=5, dilation=5, bias=False)
        self.bn_3 = nn.BatchNorm2d(planes)
        self.conv_4 = nn.Conv2d(planes * 4, planes, kernel_size=1, bias=False)
        self.bn_4 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out_1 = self.conv_1(x)
        out_1 = self.bn_1(out_1)
        out_1 = self.relu(out_1)
        out_2 = self.conv_2(x)
        out_2 = self.bn_2(out_2)
        out_2 = self.relu(out_2)
        out_3 = self.conv_3(x)
        out_3 = self.bn_3(out_3)
        out_3 = self.relu(out_3)
        out = torch.cat((out_1, out_2, out_3, residual), 1)
        out = self.conv_4(out)
        out = self.bn_4(out)
        out = self.relu(out)
        return out


@BACKBONES.register_module()
class HRNet(nn.Module):
    """HRNet backbone.

    High-Resolution Representations for Labeling Pixels and Regions
    arXiv: https://arxiv.org/abs/1904.04514

    Args:
        extra (dict): detailed configuration for each stage of HRNet.
        in_channels (int): Number of input image channels. Normally 3.
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.

    Example:
        >>> from mmseg.models import HRNet
        >>> import torch
        >>> extra = dict(
        >>>     stage1=dict(
        >>>         num_modules=1,
        >>>         num_branches=1,
        >>>         block='BOTTLENECK',
        >>>         num_blocks=(4, ),
        >>>         num_channels=(64, )),
        >>>     stage2=dict(
        >>>         num_modules=1,
        >>>         num_branches=2,
        >>>         block='BASIC',
        >>>         num_blocks=(4, 4),
        >>>         num_channels=(32, 64)),
        >>>     stage3=dict(
        >>>         num_modules=4,
        >>>         num_branches=3,
        >>>         block='BASIC',
        >>>         num_blocks=(4, 4, 4),
        >>>         num_channels=(32, 64, 128)),
        >>>     stage4=dict(
        >>>         num_modules=3,
        >>>         num_branches=4,
        >>>         block='BASIC',
        >>>         num_blocks=(4, 4, 4, 4),
        >>>         num_channels=(32, 64, 128, 256)))
        >>> self = HRNet(extra, in_channels=1)
        >>> self.eval()
        >>> inputs = torch.rand(1, 1, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 32, 8, 8)
        (1, 64, 4, 4)
        (1, 128, 2, 2)
        (1, 256, 1, 1)
    """

    blocks_dict = {
        'BASIC': BasicBlock,
        'BOTTLENECK': Bottleneck,
        'MIXEDDILATEDBLOCK': MixedDilatedBlock
    }

    def __init__(self,
                 extra,
                 in_channels=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=False):
        super(HRNet, self).__init__()
        self.extra = extra
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual

        # stem net
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, 64, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(self.norm_cfg, 64, postfix=2)

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)

        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            self.conv_cfg,
            64,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)

        self.add_module(self.norm2_name, norm2)
        self.relu = nn.ReLU(inplace=True)

        # stage 1
        self.stage1_cfg = self.extra['stage1']
        num_channels = self.stage1_cfg['num_channels'][0]
        block_type = self.stage1_cfg['block']
        num_blocks = self.stage1_cfg['num_blocks'][0]

        block = self.blocks_dict[block_type]
        stage1_out_channels = num_channels * block.expansion
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)

        # stage 2
        self.stage2_cfg = self.extra['stage2']
        num_channels = self.stage2_cfg['num_channels']
        block_type = self.stage2_cfg['block']

        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition1 = self._make_transition_layer([stage1_out_channels],
                                                       num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        # stage 3
        self.stage3_cfg = self.extra['stage3']
        num_channels = self.stage3_cfg['num_channels']
        block_type = self.stage3_cfg['block']

        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition2 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        # stage 4
        self.stage4_cfg = self.extra['stage4']
        num_channels = self.stage4_cfg['num_channels']
        block_type = self.stage4_cfg['block']

        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition3 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels)

        # 添加MixedDilatedBlock层
        self.mixed_dilated = nn.ModuleList()
        for i in range(3):  # 3层MixedDilatedBlock
            layer = nn.ModuleList()
            for j in range(len(pre_stage_channels)):
                layer.append(MixedDilatedBlock(pre_stage_channels[j], pre_stage_channels[j]))
            self.mixed_dilated.append(layer)

        # 添加DUC模块
        self.duc = nn.ModuleList()
        for i in range(len(pre_stage_channels)-1):
            self.duc.append(DUC(pre_stage_channels[i+1], pre_stage_channels[i+1]*2))

        # 添加SELayer - 使用pre_stage_channels而不是in_channels
        self.se_layers = nn.ModuleList([SELayer(pre_stage_channels[i]) for i in range(len(pre_stage_channels))])

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: the normalization layer named "norm2" """
        return getattr(self, self.norm2_name)

    def _make_transition_layer(self, num_channels_pre_layer,
                               num_channels_cur_layer):
        """Make transition layer."""
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False),
                            build_norm_layer(self.norm_cfg,
                                             num_channels_cur_layer[i])[1],
                            nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else in_channels
                    conv_downsamples.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels,
                                out_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=False),
                            build_norm_layer(self.norm_cfg, out_channels)[1],
                            nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv_downsamples))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        """Make each layer."""
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(self.norm_cfg, planes * block.expansion)[1])

        layers = []
        layers.append(
            block(
                inplanes,
                planes,
                stride,
                downsample=downsample,
                with_cp=self.with_cp,
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, in_channels, multiscale_output=True):
        """Make each stage."""
        num_modules = layer_config['num_modules']
        num_branches = layer_config['num_branches']
        num_blocks = layer_config['num_blocks']
        num_channels = layer_config['num_channels']
        block = self.blocks_dict[layer_config['block']]

        hr_modules = []
        for i in range(num_modules):
            # multi_scale_output is only used for the last module
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True

            hr_modules.append(
                HRModule(
                    num_branches,
                    block,
                    num_blocks,
                    in_channels,
                    num_channels,
                    reset_multiscale_output,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg))

        return nn.Sequential(*hr_modules), in_channels

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['num_branches']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['num_branches']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['num_branches']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        # 应用MixedDilatedBlock
        for i in range(3):
            for j in range(len(y_list)):
                y_list[j] = self.mixed_dilated[i][j](y_list[j])

        # 使用DUC进行上采样和融合
        for i in range(len(y_list)-1, 0, -1):
            y_list[i] = self.duc[i-1](y_list[i])
            y_list[i] = F.interpolate(y_list[i], size=y_list[i-1].shape[2:], mode='bilinear', align_corners=False)
            y_list[i-1] = y_list[i-1] + y_list[i]

        # 只返回最高分辨率路径
        return [y_list[0]]

    def train(self, mode=True):
        """Convert the model into training mode whill keeping the normalization
        layer freezed."""
        super(HRNet, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
