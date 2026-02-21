#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查推理是否完整
"""
import pickle as pkl
import os
import sys

def check_completeness():
    pkl_file = 'tuiliBIoU_biou_0_0/refined.pkl'
    patch_dir = 'tuiliBIoU_biou_0_0/patches/img_dir/val'
    
    # 1. 检查PKL文件
    print("Checking PKL file...")
    with open(pkl_file, 'rb') as f:
        _res = pkl.load(f)
    img_infos, masks = _res
    
    print(f"PKL contains {len(img_infos)} img_infos")
    print(f"PKL contains {len(masks)} masks")
    
    if len(img_infos) != len(masks):
        print(f"⚠️  WARNING: img_infos ({len(img_infos)}) != masks ({len(masks)})")
    else:
        print("✓ img_infos and masks count match")
    print()
    
    # 2. 检查实际文件数
    print("Checking actual patch files...")
    if os.path.exists(patch_dir):
        actual_files = [f for f in os.listdir(patch_dir) if f.endswith('.png')]
        print(f"Actual patch files: {len(actual_files)}")
    else:
        print(f"⚠️  Patch directory not found: {patch_dir}")
        return
    print()
    
    # 3. 比较
    print("Comparison:")
    print(f"  Actual files: {len(actual_files)}")
    print(f"  PKL results: {len(img_infos)}")
    print(f"  Missing: {len(actual_files) - len(img_infos)}")
    print(f"  Coverage: {len(img_infos) / len(actual_files) * 100:.2f}%")
    print()
    
    # 4. 检查PKL中的filename是否都在实际文件中
    print("Checking if PKL filenames exist in actual files...")
    pkl_filenames = set([info['filename'] for info in img_infos if 'filename' in info])
    actual_filenames = set(actual_files)
    
    pkl_only = pkl_filenames - actual_filenames
    actual_only = actual_filenames - pkl_filenames
    common = pkl_filenames & actual_filenames
    
    print(f"  Common: {len(common)}")
    print(f"  PKL only: {len(pkl_only)}")
    print(f"  Actual only: {len(actual_only)}")
    print()
    
    if pkl_only:
        print(f"⚠️  WARNING: {len(pkl_only)} filenames in PKL but not in actual files")
        print(f"  Sample: {list(pkl_only)[:10]}")
        print()
    
    if actual_only:
        print(f"⚠️  WARNING: {len(actual_only)} filenames in actual files but not in PKL")
        print(f"  These patches were not processed during inference!")
        print(f"  Sample: {list(actual_only)[:20]}")
        print()
    
    # 5. 检查是否有模式
    if actual_only:
        print("Analyzing missing files pattern...")
        missing_pids = [int(f.replace('.png', '')) for f in list(actual_only)[:100] if f.replace('.png', '').isdigit()]
        if missing_pids:
            missing_pids.sort()
            print(f"  Missing PID range: {min(missing_pids)} - {max(missing_pids)}")
            print(f"  Sample missing PIDs: {missing_pids[:20]}")
            
            # 检查是否连续
            gaps = []
            for i in range(len(missing_pids) - 1):
                if missing_pids[i+1] - missing_pids[i] > 1:
                    gaps.append((missing_pids[i], missing_pids[i+1]))
            if gaps:
                print(f"  Found {len(gaps)} gaps in missing PIDs")
                print(f"  Sample gaps: {gaps[:5]}")
            else:
                print("  Missing PIDs appear to be continuous")

if __name__ == '__main__':
    check_completeness()

