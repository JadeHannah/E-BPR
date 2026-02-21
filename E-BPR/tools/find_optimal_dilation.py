#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试不同的dilation值，找到最适合小补丁的dilation参数
"""
import os
import sys
import json
import numpy as np
from PIL import Image
import cv2
from pathlib import Path

# 添加tools目录到路径
sys.path.insert(0, os.path.dirname(__file__))
from split_patches import (
    get_dets, compute_biou, cal_iou, query_gt_mask,
    mask_to_boundary, crop
)
from pycocotools.coco import COCO


def test_dilation_values(coarse_json, gt_json, img_dir, inst_id, 
                        dilation_ratios=[0.001, 0.002, 0.003, 0.005, 0.01, 0.02],
                        fixed_dilations=[1, 2, 3, 4, 5],
                        iou_thresh=0.3):
    """测试不同的dilation值"""
    
    # 加载数据
    coco = COCO(gt_json)
    with open(coarse_json, 'r') as f:
        coarse_data = json.load(f)
    
    annotations = coarse_data.get('annotations', [])
    if inst_id >= len(annotations):
        print(f"Error: inst_id {inst_id} >= {len(annotations)}")
        return
    
    inst = annotations[inst_id]
    imgid = inst['image_id']
    catid = inst['category_id']
    imgname = coco.imgs[imgid]["file_name"]
    seg_path = inst.get('segmentation', '')
    
    print(f"\n{'='*80}")
    print(f"Testing dilation values for annotation {inst_id}")
    print(f"{'='*80}")
    print(f"Image: {imgname}")
    print(f"Segmentation: {seg_path}")
    
    # 加载mask
    img_path = os.path.join(img_dir, imgname)
    if not os.path.exists(img_path) or not os.path.exists(seg_path):
        print("✗ Files not found")
        return
    
    mask_raw = np.array(Image.open(seg_path))
    if mask_raw.ndim == 3:
        mask_raw = mask_raw[:, :, 0]
    maskdt = (mask_raw > 0).astype(np.uint8)
    
    if maskdt.sum() == 0:
        print("✗ Empty mask")
        return
    
    # 查询GT mask
    maskgt = query_gt_mask(maskdt, coco, imgid, catid, use_biou=False)
    
    if maskgt.sum() == 0:
        print("✗ No GT mask found")
        return
    
    # 检测补丁位置
    dets = get_dets(maskdt, patch_size=64, iou_thresh=0.3)
    
    if len(dets) == 0:
        print("✗ No patches detected")
        return
    
    print(f"\nFound {len(dets)} patches, testing first patch...")
    
    # 裁剪第一个补丁
    img = cv2.imread(img_path)
    img_patches, dt_patches, gt_patches = crop(img, maskdt, maskgt, dets[:1], padding=0)
    
    dt_patch = dt_patches[0].astype(np.uint8)
    gt_patch = gt_patches[0].astype(np.uint8)
    
    print(f"\nPatch info:")
    print(f"  Shape: {dt_patch.shape}")
    print(f"  DT mask pixels: {dt_patch.sum()}")
    print(f"  GT mask pixels: {gt_patch.sum()}")
    
    # 计算IoU作为参考
    iou = cal_iou(dt_patch, gt_patch)
    print(f"  IoU: {iou:.4f}")
    
    print(f"\n{'='*80}")
    print(f"Testing dilation_ratio values:")
    print(f"{'='*80}")
    print(f"{'dilation_ratio':<20} {'dilation':<10} {'pred_boundary':<15} {'gt_boundary':<15} "
          f"{'inter_b':<10} {'union_b':<10} {'BIoU':<10} {'final_iou':<10} {'pass':<10}")
    print("-" * 80)
    
    best_ratio = None
    best_biou = 0
    best_final = 0
    
    for dilation_ratio in dilation_ratios:
        # 计算实际dilation值
        h, w = dt_patch.shape
        diag_len = np.sqrt(h ** 2 + w ** 2)
        dilation = max(1, int(round(dilation_ratio * diag_len)))
        
        # 对于小补丁，限制dilation
        if h <= 64 and w <= 64:
            dilation = max(1, min(2, dilation))
        
        # 确保为奇数
        if dilation % 2 == 0:
            dilation += 1
        
        # 计算边界
        pred_boundary = mask_to_boundary(dt_patch, dilation_ratio)
        gt_boundary = mask_to_boundary(gt_patch, dilation_ratio)
        
        pred_boundary_pixels = pred_boundary.sum()
        gt_boundary_pixels = gt_boundary.sum()
        inter_b = np.logical_and(pred_boundary, gt_boundary).sum()
        union_b = np.logical_or(pred_boundary, gt_boundary).sum()
        
        biou = inter_b / union_b if union_b > 0 else 0.0
        final_iou = min(iou, biou)
        pass_thresh = final_iou >= iou_thresh
        
        print(f"{dilation_ratio:<20.4f} {dilation:<10} {pred_boundary_pixels:<15} "
              f"{gt_boundary_pixels:<15} {inter_b:<10} {union_b:<10} "
              f"{biou:<10.4f} {final_iou:<10.4f} {'✓' if pass_thresh else '✗':<10}")
        
        if biou > best_biou:
            best_biou = biou
            best_final = final_iou
            best_ratio = dilation_ratio
    
    print(f"\n{'='*80}")
    print(f"Testing fixed dilation values:")
    print(f"{'='*80}")
    print(f"{'fixed_dilation':<20} {'pred_boundary':<15} {'gt_boundary':<15} "
          f"{'inter_b':<10} {'union_b':<10} {'BIoU':<10} {'final_iou':<10} {'pass':<10}")
    print("-" * 80)
    
    best_fixed = None
    best_fixed_biou = 0
    best_fixed_final = 0
    
    for fixed_dilation in fixed_dilations:
        # 手动设置dilation（忽略dilation_ratio）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fixed_dilation, fixed_dilation))
        pred_boundary = cv2.dilate(dt_patch, kernel) - cv2.erode(dt_patch, kernel)
        gt_boundary = cv2.dilate(gt_patch, kernel) - cv2.erode(gt_patch, kernel)
        
        pred_boundary_pixels = pred_boundary.sum()
        gt_boundary_pixels = gt_boundary.sum()
        inter_b = np.logical_and(pred_boundary, gt_boundary).sum()
        union_b = np.logical_or(pred_boundary, gt_boundary).sum()
        
        biou = inter_b / union_b if union_b > 0 else 0.0
        final_iou = min(iou, biou)
        pass_thresh = final_iou >= iou_thresh
        
        print(f"{fixed_dilation:<20} {pred_boundary_pixels:<15} "
              f"{gt_boundary_pixels:<15} {inter_b:<10} {union_b:<10} "
              f"{biou:<10.4f} {final_iou:<10.4f} {'✓' if pass_thresh else '✗':<10}")
        
        if biou > best_fixed_biou:
            best_fixed_biou = biou
            best_fixed_final = final_iou
            best_fixed = fixed_dilation
    
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"{'='*80}")
    print(f"Best dilation_ratio: {best_ratio:.4f} (BIoU={best_biou:.4f}, final={best_final:.4f})")
    print(f"Best fixed_dilation: {best_fixed} (BIoU={best_fixed_biou:.4f}, final={best_fixed_final:.4f})")
    print(f"\nRecommendation:")
    if best_fixed_biou > best_biou:
        print(f"  Use fixed_dilation={best_fixed} for small patches (64x64)")
        print(f"  This gives BIoU={best_fixed_biou:.4f} vs ratio-based {best_biou:.4f}")
    else:
        print(f"  Use dilation_ratio={best_ratio:.4f}")
        print(f"  This gives BIoU={best_biou:.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Find optimal dilation for BIoU')
    parser.add_argument('coarse_json', help='Coarse JSON file')
    parser.add_argument('gt_json', help='GT JSON file')
    parser.add_argument('img_dir', help='Image directory')
    parser.add_argument('--inst-id', type=int, default=0, help='Annotation ID to test')
    parser.add_argument('--dilation-ratios', type=float, nargs='+', 
                       default=[0.001, 0.002, 0.003, 0.005, 0.01, 0.02],
                       help='Dilation ratios to test')
    parser.add_argument('--fixed-dilations', type=int, nargs='+',
                       default=[1, 2, 3, 4, 5],
                       help='Fixed dilation values to test')
    parser.add_argument('--iou-thresh', type=float, default=0.3, help='IoU threshold')
    
    args = parser.parse_args()
    
    test_dilation_values(
        args.coarse_json,
        args.gt_json,
        args.img_dir,
        args.inst_id,
        args.dilation_ratios,
        args.fixed_dilations,
        args.iou_thresh
    )


if __name__ == '__main__':
    main()

