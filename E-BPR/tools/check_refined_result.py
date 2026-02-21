#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查refined结果，诊断为什么BIoU没有起作用
"""
import json
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools import mask as maskUtils

def check_refined_result(result_dir, image_name=None):
    """
    检查refined结果
    
    Args:
        result_dir: 结果目录（如 tuiliBIoU_biou_0_200）
        image_name: 图像名称（如 frankfurt_000000_000294），如果为None则检查所有
    """
    result_path = Path(result_dir)
    
    # 检查文件
    coarse_json = result_path / 'coarse.json'
    refined_json = result_path / 'refined.json'
    patches_dir = result_path / 'patches' / 'detail_dir' / 'val'
    
    if not coarse_json.exists():
        print(f"Error: coarse.json not found: {coarse_json}")
        return
    
    if not refined_json.exists():
        print(f"Error: refined.json not found: {refined_json}")
        return
    
    # 加载JSON
    with open(coarse_json, 'r') as f:
        coarse_data = json.load(f)
    
    with open(refined_json, 'r') as f:
        refined_data = json.load(f)
    
    coarse_anns = {ann['id']: ann for ann in coarse_data.get('annotations', [])}
    refined_anns = {ann['id']: ann for ann in refined_data.get('annotations', [])}
    
    print(f"\n{'='*80}")
    print(f"Checking refined results in: {result_dir}")
    print(f"{'='*80}")
    print(f"Coarse annotations: {len(coarse_anns)}")
    print(f"Refined annotations: {len(refined_anns)}")
    
    # 检查detail文件
    if patches_dir.exists():
        detail_files = list(patches_dir.glob('*.txt'))
        print(f"Detail files: {len(detail_files)}")
    else:
        print(f"Detail directory not found: {patches_dir}")
        # 检查patches目录是否存在
        patches_base = result_path / 'patches'
        if patches_base.exists():
            print(f"Patches base directory exists: {patches_base}")
            print(f"Contents: {list(patches_base.iterdir())}")
        else:
            print(f"Patches base directory not found: {patches_base}")
        detail_files = []
    
    # 检查特定图像或随机选择几个
    if image_name:
        # 找到该图像的所有annotations
        image_id = None
        for img in coarse_data.get('images', []):
            if image_name in img.get('file_name', ''):
                image_id = img['id']
                break
        
        if image_id is None:
            print(f"Error: Image {image_name} not found")
            return
        
        print(f"\nChecking image_id={image_id} (image_name contains '{image_name}')")
        image_anns = [ann for ann in coarse_anns.values() if ann['image_id'] == image_id]
    else:
        # 随机选择前5个
        image_anns = list(coarse_anns.values())[:5]
        print(f"\nChecking first 5 annotations")
    
    print(f"\n{'='*80}")
    print(f"Annotation Comparison:")
    print(f"{'='*80}")
    print(f"{'ID':<8} {'Image_ID':<10} {'Has_Detail':<12} {'In_Refined':<12} {'Seg_Type':<15} {'Mask_Empty':<12}")
    print("-" * 80)
    
    for ann in image_anns:
        ann_id = ann['id']
        has_detail = (patches_dir / f"{ann_id}.txt").exists() if patches_dir.exists() else False
        in_refined = ann_id in refined_anns
        
        if in_refined:
            refined_ann = refined_anns[ann_id]
            seg = refined_ann.get('segmentation', {})
            if isinstance(seg, dict) and 'counts' in seg:
                # RLE格式
                try:
                    mask = maskUtils.decode(seg)
                    mask_empty = mask.sum() == 0
                    seg_type = "RLE"
                except:
                    mask_empty = True
                    seg_type = "RLE(Error)"
            elif isinstance(seg, str):
                seg_type = "String"
                mask_empty = True
            else:
                seg_type = "Unknown"
                mask_empty = True
        else:
            seg_type = "N/A"
            mask_empty = True
        
        print(f"{ann_id:<8} {ann['image_id']:<10} {str(has_detail):<12} {str(in_refined):<12} {seg_type:<15} {str(mask_empty):<12}")
    
    # 统计信息
    print(f"\n{'='*80}")
    print(f"Statistics:")
    print(f"{'='*80}")
    
    total_coarse = len(coarse_anns)
    total_refined = len(refined_anns)
    total_detail = len(detail_files)
    
    # 检查refined中的空mask
    empty_masks = 0
    rle_masks = 0
    string_masks = 0
    
    for ann in refined_anns.values():
        seg = ann.get('segmentation', {})
        if isinstance(seg, dict) and 'counts' in seg:
            rle_masks += 1
            try:
                mask = maskUtils.decode(seg)
                if mask.sum() == 0:
                    empty_masks += 1
            except:
                empty_masks += 1
        elif isinstance(seg, str):
            string_masks += 1
    
    print(f"Total coarse annotations: {total_coarse}")
    print(f"Total refined annotations: {total_refined}")
    print(f"Total detail files: {total_detail}")
    print(f"RLE format masks: {rle_masks}")
    print(f"String format masks: {string_masks}")
    print(f"Empty masks in refined: {empty_masks}")
    print(f"Annotations with detail files: {total_detail}/{total_coarse} ({100*total_detail/total_coarse:.1f}%)")
    
    # 诊断
    print(f"\n{'='*80}")
    print(f"Diagnosis:")
    print(f"{'='*80}")
    
    if total_detail == 0:
        print("❌ No detail files found - all patches were filtered out!")
        print("   → BIoU threshold may be too high")
        print("   → Check split_patches.py output for filtering statistics")
    elif total_detail < total_coarse * 0.1:
        print(f"⚠️  Very few detail files ({total_detail}/{total_coarse})")
        print("   → Most patches were filtered out by BIoU threshold")
        print("   → Consider lowering the BIoU threshold")
    else:
        print(f"✓ Detail files exist ({total_detail}/{total_coarse})")
    
    if empty_masks > 0:
        print(f"⚠️  {empty_masks} empty masks in refined.json")
        print("   → These annotations may not have been properly refined")
    
    if string_masks > 0:
        print(f"⚠️  {string_masks} string format masks (should be RLE)")
        print("   → merge_patches.py may not have processed these correctly")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Check refined results')
    parser.add_argument('result_dir', help='Result directory (e.g., tuiliBIoU_biou_0_200)')
    parser.add_argument('--image-name', type=str, default=None,
                       help='Image name to check (e.g., frankfurt_000000_000294)')
    
    args = parser.parse_args()
    
    check_refined_result(args.result_dir, args.image_name)


if __name__ == '__main__':
    main()

