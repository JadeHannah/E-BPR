import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder


@SEGMENTORS.register_module()
class EncoderDecoderRefine(EncoderDecoder):
    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 use_coarse_mask=True,      # whether to use coarse mask as input
                 output_float=False,        # whether to return float instead of binary mask
            ):
        super(EncoderDecoderRefine, self).__init__(
            backbone,
            decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained
        )
        self.use_coarse_mask = use_coarse_mask
        self.output_float = output_float

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def forward_train(self, img, img_metas, gt_semantic_seg, coarse_mask):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if self.use_coarse_mask:
            coarse_mask = (coarse_mask - 0.5) / 0.5
            img = torch.cat([img, coarse_mask[:,None,...]], dim=1)
        x = self.extract_feat(img)

        losses = dict()
        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses

    def simple_test(self, img, img_meta, coarse_mask, rescale=True):
        if self.use_coarse_mask:
            coarse_mask = (coarse_mask[0] - 0.5) / 0.5
            img = torch.cat([img, coarse_mask[:,None,...]], dim=1)
        # res = super().simple_test(img, img_meta, rescale)

        seg_logit = self.inference(img, img_meta, rescale)
        
        # Handle different output formats
        if self.output_float:
            seg_pred = seg_logit[:,1,:,:]
        else:
            # Standard multi-class segmentation: use argmax
            # For num_classes=2: argmax returns 0 (background) or 1 (foreground)
            seg_pred = seg_logit.argmax(dim=1)
        
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)

        return seg_pred
    
    def aug_test(self, imgs, img_metas, coarse_mask, rescale=True):
        """Test with augmentations for binary segmentation."""
        assert rescale
        # Handle coarse mask for augmented images
        if self.use_coarse_mask:
            coarse_mask_list = []
            for i in range(len(imgs)):
                if isinstance(coarse_mask, list):
                    cm = coarse_mask[i] if i < len(coarse_mask) else coarse_mask[0]
                else:
                    cm = coarse_mask
                cm = (cm - 0.5) / 0.5
                img_with_mask = torch.cat([imgs[i], cm[:,None,...]], dim=1)
                coarse_mask_list.append(img_with_mask)
            imgs = coarse_mask_list
        
        # Get augmented seg logit
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        
        # Standard multi-class segmentation: use argmax
        seg_pred = seg_logit.argmax(dim=1)
        
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
