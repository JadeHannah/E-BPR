#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
调试IoU计算问题
检查为什么很多IoU都是0
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from split_patches import cal_iou, compute_biou, crop, query_gt_mask, get_dets
from pycocotools.coco import COCO
from PIL import Image
import cv2
import json

def debug_iou_for_patch(coarse_json, gt_json, img_dir, inst_id):
    """调试单个annotation的IoU计算"""
    
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
    print(f"Debugging IoU calculation for annotation {inst_id}")
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
    
    print(f"\nOriginal mask info:")
    print(f"  Shape: {maskdt.shape}")
    print(f"  Non-zero pixels: {maskdt.sum()}")
    print(f"  Unique values: {np.unique(maskdt)}")
    
    # 查询GT mask
    maskgt = query_gt_mask(maskdt, coco, imgid, catid, use_biou=False)
    
    print(f"\nGT mask info:")
    print(f"  Non-zero pixels: {maskgt.sum()}")
    print(f"  Unique values: {np.unique(maskgt)}")
    
    if maskgt.sum() == 0:
        print("✗ No GT mask found")
        return
    
    # 检测补丁位置
    dets = get_dets(maskdt, patch_size=64, iou_thresh=0.3)
    
    print(f"\nDetected patches: {len(dets)}")
    
    if len(dets) == 0:
        print("✗ No patches detected")
        return
    
    # 裁剪补丁
    img = cv2.imread(img_path)
    img_patches, dt_patches, gt_patches = crop(img, maskdt, maskgt, dets[:1], padding=0)
    
    dt_patch = dt_patches[0]
    gt_patch = gt_patches[0]
    
    print(f"\n{'='*80}")
    print(f"Patch Analysis (first patch):")
    print(f"{'='*80}")
    print(f"Patch shape: {dt_patch.shape}")
    print(f"DT patch:")
    print(f"  Dtype: {dt_patch.dtype}")
    print(f"  Unique values: {np.unique(dt_patch)}")
    print(f"  Non-zero pixels: {dt_patch.sum()}")
    print(f"  Min/Max: {dt_patch.min()}/{dt_patch.max()}")
    
    print(f"\nGT patch:")
    print(f"  Dtype: {gt_patch.dtype}")
    print(f"  Unique values: {np.unique(gt_patch)}")
    print(f"  Non-zero pixels: {gt_patch.sum()}")
    print(f"  Min/Max: {gt_patch.min()}/{gt_patch.max()}")
    
    # 检查重叠
    intersection = np.logical_and(dt_patch > 0, gt_patch > 0).sum()
    union = np.logical_or(dt_patch > 0, gt_patch > 0).sum()
    
    print(f"\nOverlap analysis:")
    print(f"  Intersection (both > 0): {intersection}")
    print(f"  Union (either > 0): {union}")
    print(f"  IoU (manual): {intersection / union if union > 0 else 0:.6f}")
    
    # 使用cal_iou计算
    dt_binary = (dt_patch > 0).astype(np.uint8)
    gt_binary = (gt_patch > 0).astype(np.uint8)
    
    iou_cal = cal_iou(dt_binary, gt_binary)
    print(f"  IoU (cal_iou): {iou_cal:.6f}")
    
    # 使用compute_biou计算
    final_iou, iou_biou, biou = compute_biou(dt_binary, gt_binary, 
                                             dilation_ratio=0.005,
                                             combine_with_iou=False)
    print(f"  IoU (from compute_biou): {iou_biou:.6f}")
    print(f"  BIoU: {biou:.6f}")
    print(f"  Final (BIoU only): {final_iou:.6f}")
    
    # 检查问题
    print(f"\n{'='*80}")
    print(f"Diagnosis:")
    print(f"{'='*80}")
    
    if dt_patch.sum() == 0:
        print("❌ DT patch is empty!")
    elif gt_patch.sum() == 0:
        print("❌ GT patch is empty!")
    elif intersection == 0:
        print("❌ No overlap between DT and GT patches!")
        print("   This is why IoU is 0 or very low")
    elif union == 0:
        print("❌ Union is 0 (should not happen)")
    else:
        print(f"✓ Patches have overlap: {intersection}/{union} = {intersection/union:.4f}")
        if iou_cal < 0.1:
            print(f"⚠️  IoU is very low ({iou_cal:.4f}), but patches do overlap")
            print("   This might be normal if the overlap is small relative to union")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Debug IoU calculation')
    parser.add_argument('coarse_json', help='Coarse JSON file')
    parser.add_argument('gt_json', help='GT JSON file')
    parser.add_argument('img_dir', help='Image directory')
    parser.add_argument('--inst-id', type=int, default=0, help='Annotation ID to test')
    
    args = parser.parse_args()
    
    debug_iou_for_patch(args.coarse_json, args.gt_json, args.img_dir, args.inst_id)


if __name__ == '__main__':
    main()

