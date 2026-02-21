#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
调试补丁生成问题
检查为什么patches文件夹是空的
"""
import os
import json
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

# 导入split_patches中的函数
import sys
sys.path.insert(0, os.path.dirname(__file__))
from split_patches import compute_biou, cal_iou, query_gt_mask
from pycocotools.coco import COCO


def check_patches(result_dir, gt_json, iou_thresh, use_biou=False, biou_dilation_ratio=0.005):
    """
    检查补丁生成情况
    """
    result_path = Path(result_dir)
    
    # 检查coarse.json
    coarse_json = result_path / 'coarse.json'
    if not coarse_json.exists():
        print(f"Error: coarse.json not found: {coarse_json}")
        return
    
    with open(coarse_json, 'r') as f:
        coarse_data = json.load(f)
    
    annotations = coarse_data.get('annotations', [])
    print(f"Total annotations in coarse.json: {len(annotations)}")
    
    # 检查patches目录
    patches_dir = result_path / 'patches'
    img_dir = patches_dir / 'img_dir' / 'val'
    detail_dir = patches_dir / 'detail_dir' / 'val'
    
    print(f"\nPatches directory structure:")
    print(f"  img_dir exists: {img_dir.exists()}")
    print(f"  detail_dir exists: {detail_dir.exists()}")
    
    if img_dir.exists():
        patch_files = list(img_dir.glob('*.png'))
        print(f"  Number of patch images: {len(patch_files)}")
        if len(patch_files) > 0:
            print(f"  Example patch files: {patch_files[:5]}")
    else:
        print(f"  img_dir does not exist!")
    
    if detail_dir.exists():
        detail_files = list(detail_dir.glob('*.txt'))
        print(f"  Number of detail files: {len(detail_files)}")
        if len(detail_files) > 0:
            print(f"  Example detail files: {detail_files[:5]}")
    else:
        print(f"  detail_dir does not exist!")
    
    # 检查前几个annotation的补丁情况
    print(f"\nChecking first 5 annotations:")
    coco = COCO(gt_json)
    
    for i, ann in enumerate(annotations[:5]):
        inst_id = ann['id']
        img_id = ann['image_id']
        seg_path = ann.get('segmentation', '')
        
        print(f"\n  Annotation {i+1} (id={inst_id}, image_id={img_id}):")
        print(f"    Segmentation path: {seg_path}")
        
        # 检查detail文件
        detail_file = detail_dir / f"{inst_id}.txt"
        if detail_file.exists():
            with open(detail_file, 'r') as f:
                lines = f.readlines()
            print(f"    Detail file exists: {len(lines)} patches")
            if len(lines) > 0:
                print(f"    First line: {lines[0].strip()}")
        else:
            print(f"    Detail file NOT found: {detail_file}")
            
            # 尝试分析为什么没有补丁
            if os.path.exists(seg_path):
                mask = np.array(Image.open(seg_path).convert('L')) > 0
                mask = mask.astype(np.uint8)
                
                # 查询GT mask
                cat_id = ann['category_id']
                gt_mask = query_gt_mask(mask, coco, img_id, cat_id, 
                                      use_biou=use_biou, 
                                      biou_dilation_ratio=biou_dilation_ratio)
                
                # 计算一个示例补丁的分数
                if mask.sum() > 0 and gt_mask.sum() > 0:
                    # 取mask的中心区域作为示例
                    h, w = mask.shape
                    center_y, center_x = h // 2, w // 2
                    patch_size = 64
                    y1 = max(0, center_y - patch_size // 2)
                    y2 = min(h, center_y + patch_size // 2)
                    x1 = max(0, center_x - patch_size // 2)
                    x2 = min(w, center_x + patch_size // 2)
                    
                    dt_patch = mask[y1:y2, x1:x2]
                    gt_patch = gt_mask[y1:y2, x1:x2]
                    
                    if dt_patch.sum() > 0 and gt_patch.sum() > 0:
                        if use_biou:
                            final_iou, iou, biou = compute_biou(dt_patch, gt_patch, 
                                                              biou_dilation_ratio, 
                                                              combine_with_iou=True)
                            print(f"    Example patch score: final={final_iou:.4f}, iou={iou:.4f}, biou={biou:.4f}")
                            print(f"    Threshold: {iou_thresh:.3f}, Pass: {final_iou >= iou_thresh}")
                        else:
                            score = cal_iou(dt_patch, gt_patch)
                            print(f"    Example patch IoU: {score:.4f}")
                            print(f"    Threshold: {iou_thresh:.3f}, Pass: {score >= iou_thresh}")


def main():
    parser = argparse.ArgumentParser(description='Debug patch generation')
    parser.add_argument('result_dir', help='Result directory (e.g., tuiliBIoU_biou_0_300)')
    parser.add_argument('gt_json', help='GT JSON file path')
    parser.add_argument('--iou-thresh', type=float, default=0.3, help='IoU/BIoU threshold')
    parser.add_argument('--use-biou', action='store_true', help='Use BIoU')
    parser.add_argument('--biou-dilation-ratio', type=float, default=0.005, help='BIoU dilation ratio')
    
    args = parser.parse_args()
    
    check_patches(args.result_dir, args.gt_json, args.iou_thresh, 
                  args.use_biou, args.biou_dilation_ratio)


if __name__ == '__main__':
    main()

