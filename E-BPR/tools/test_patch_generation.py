#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试补丁生成，检查为什么没有生成补丁
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
    find_float_boundary
)
from pycocotools.coco import COCO
import torch
import torch.nn.functional as F


def test_single_annotation(coarse_json, gt_json, img_dir, inst_id, 
                          iou_thresh=0.3, use_biou=False, biou_dilation_ratio=0.005,
                          patch_size=64):
    """测试单个annotation的补丁生成"""
    
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
    
    print(f"\n{'='*60}")
    print(f"Testing annotation {inst_id}")
    print(f"{'='*60}")
    print(f"Image ID: {imgid}")
    print(f"Category ID: {catid}")
    print(f"Image name: {imgname}")
    print(f"Segmentation path: {seg_path}")
    
    # 检查文件是否存在
    img_path = os.path.join(img_dir, imgname)
    if not os.path.exists(img_path):
        print(f"✗ Image not found: {img_path}")
        return
    
    if not os.path.exists(seg_path):
        print(f"✗ Segmentation mask not found: {seg_path}")
        return
    
    # 加载mask
    mask_raw = np.array(Image.open(seg_path))
    if mask_raw.ndim == 3:
        mask_raw = mask_raw[:, :, 0]
    maskdt = (mask_raw > 0).astype(np.uint8)
    
    print(f"\nMask info:")
    print(f"  Shape: {maskdt.shape}")
    print(f"  Non-zero pixels: {maskdt.sum()}")
    print(f"  Ratio: {maskdt.sum() / maskdt.size:.4f}")
    
    if maskdt.sum() == 0:
        print("✗ Empty mask, cannot generate patches")
        return
    
    # 查询GT mask
    print(f"\nQuerying GT mask...")
    print(f"  Image ID: {imgid}")
    print(f"  Category ID: {catid}")
    
    maskgt = query_gt_mask(maskdt, coco, imgid, catid, 
                          use_biou=use_biou, 
                          biou_dilation_ratio=biou_dilation_ratio,
                          debug=True)
    
    print(f"\nGT mask info:")
    print(f"  Non-zero pixels: {maskgt.sum()}")
    print(f"  Ratio: {maskgt.sum() / maskgt.size:.4f}")
    
    if maskgt.sum() == 0:
        print("✗ No matching GT mask found")
        return
    
    # 检测补丁位置
    print(f"\nDetecting patch locations...")
    print(f"  Patch size: {patch_size}")
    print(f"  NMS IoU threshold: {iou_thresh}")
    
    dets = get_dets(maskdt, patch_size, iou_thresh)
    
    print(f"  Detected patches: {len(dets)}")
    
    if len(dets) == 0:
        print("\n✗ No patches detected!")
        print("  Possible reasons:")
        print("  1. Mask too small or sparse")
        print("  2. Float boundary detection failed")
        print("  3. NMS filtered out all detections")
        
        # 检查float boundary
        fbmask = find_float_boundary(maskdt, width=3)
        boundary_pixels = np.sum(fbmask > 0)
        print(f"\n  Float boundary pixels: {boundary_pixels}")
        if boundary_pixels == 0:
            print("  ✗ No boundary pixels detected!")
        return
    
    print(f"  ✓ Found {len(dets)} patch locations")
    
    # 测试前几个补丁的BIoU/IoU分数
    print(f"\nTesting patch scores (first 5 patches):")
    img = cv2.imread(img_path)
    if img is None:
        print("✗ Cannot load image")
        return
    
    # 裁剪补丁
    from split_patches import crop
    img_patches, dt_patches, gt_patches = crop(img, maskdt, maskgt, dets, padding=0)
    
    saved_count = 0
    filtered_count = 0
    scores_list = []
    
    for i in range(min(5, len(dets))):
        dt_patch = dt_patches[i].astype(np.uint8)
        gt_patch = gt_patches[i].astype(np.uint8)
        
        if use_biou:
            final_iou, iou, biou = compute_biou(dt_patch, gt_patch, 
                                              biou_dilation_ratio, 
                                              combine_with_iou=True)
            score = final_iou
            
            # 调试信息：检查边界
            from split_patches import mask_to_boundary
            pred_boundary = mask_to_boundary(dt_patch, biou_dilation_ratio)
            gt_boundary = mask_to_boundary(gt_patch, biou_dilation_ratio)
            pred_boundary_pixels = pred_boundary.sum()
            gt_boundary_pixels = gt_boundary.sum()
            inter_b = np.logical_and(pred_boundary, gt_boundary).sum()
            union_b = np.logical_or(pred_boundary, gt_boundary).sum()
            
            print(f"  Patch {i+1}: final={final_iou:.4f}, iou={iou:.4f}, biou={biou:.4f}, "
                  f"threshold={iou_thresh:.3f}, pass={final_iou >= iou_thresh}")
            print(f"    Boundary: pred={pred_boundary_pixels}, gt={gt_boundary_pixels}, "
                  f"inter={inter_b}, union={union_b}")
        else:
            score = cal_iou(dt_patch, gt_patch)
            print(f"  Patch {i+1}: iou={score:.4f}, threshold={iou_thresh:.3f}, "
                  f"pass={score >= iou_thresh}")
        
        scores_list.append(score)
        if score >= iou_thresh:
            saved_count += 1
        else:
            filtered_count += 1
    
    print(f"\nSummary:")
    print(f"  Total patches detected: {len(dets)}")
    print(f"  Would save (first 5): {saved_count}")
    print(f"  Would filter (first 5): {filtered_count}")
    if scores_list:
        print(f"  Score range: {min(scores_list):.4f} - {max(scores_list):.4f}")
        print(f"  Average score: {np.mean(scores_list):.4f}")
        if max(scores_list) < iou_thresh:
            print(f"\n⚠️  WARNING: All patch scores < threshold {iou_thresh:.3f}")
            print(f"   Consider lowering the threshold to {max(scores_list):.3f} or lower")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test patch generation for a single annotation')
    parser.add_argument('coarse_json', help='Coarse JSON file')
    parser.add_argument('gt_json', help='GT JSON file')
    parser.add_argument('img_dir', help='Image directory')
    parser.add_argument('--inst-id', type=int, default=0, help='Annotation ID to test')
    parser.add_argument('--iou-thresh', type=float, default=0.3, help='IoU/BIoU threshold')
    parser.add_argument('--use-biou', action='store_true', help='Use BIoU')
    parser.add_argument('--biou-dilation-ratio', type=float, default=0.005, help='BIoU dilation ratio')
    parser.add_argument('--patch-size', type=int, default=64, help='Patch size')
    
    args = parser.parse_args()
    
    # 设置全局args（split_patches需要）
    import split_patches
    split_patches.args = type('Args', (), {
        'use_biou': args.use_biou,
        'biou_dilation_ratio': args.biou_dilation_ratio,
        'iou_thresh': args.iou_thresh,
        'patch_size': args.patch_size,
        'max_inst': 20
    })()
    
    test_single_annotation(
        args.coarse_json,
        args.gt_json,
        args.img_dir,
        args.inst_id,
        args.iou_thresh,
        args.use_biou,
        args.biou_dilation_ratio,
        args.patch_size
    )


if __name__ == '__main__':
    main()

