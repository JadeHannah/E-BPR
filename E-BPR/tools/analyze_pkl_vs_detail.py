#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析PKL和detail文件的匹配问题
"""
import pickle as pkl
import os
import sys

def analyze():
    pkl_file = 'tuiliBIoU_biou_0_0/refined.pkl'
    detail_dir = 'tuiliBIoU_biou_0_0/patches/detail_dir/val'
    
    # 1. 加载PKL
    print("Loading PKL file...")
    with open(pkl_file, 'rb') as f:
        _res = pkl.load(f)
    img_infos, masks = _res
    print(f"PKL contains {len(img_infos)} results")
    
    # 2. 提取PKL中的patch ID
    pkl_pids = {}
    for idx, img_info in enumerate(img_infos):
        if 'filename' in img_info:
            filename = img_info['filename']
            try:
                pid = int(filename.replace('.png', ''))
                pkl_pids[pid] = idx  # 保存索引以便后续检查
            except ValueError:
                pass
    
    print(f"Extracted {len(pkl_pids)} patch IDs from PKL")
    print(f"PKL PID range: {min(pkl_pids.keys())} - {max(pkl_pids.keys())}")
    print()
    
    # 3. 加载detail文件中的patch ID
    print("Loading detail files...")
    detail_pids = set()
    detail_pid_to_inst = {}
    
    detail_files = [f for f in os.listdir(detail_dir) if f.endswith('.txt')]
    print(f"Found {len(detail_files)} detail files")
    
    for detail_file in detail_files:
        try:
            inst_id = int(detail_file.replace('.txt', ''))
        except ValueError:
            continue
        
        detail_path = os.path.join(detail_dir, detail_file)
        with open(detail_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 1:
                continue
            try:
                pid = int(parts[0])
                detail_pids.add(pid)
                detail_pid_to_inst[pid] = inst_id
            except ValueError:
                continue
    
    print(f"Extracted {len(detail_pids)} patch IDs from detail files")
    print(f"Detail PID range: {min(detail_pids)} - {max(detail_pids)}")
    print()
    
    # 4. 检查匹配
    common = set(pkl_pids.keys()) & detail_pids
    pkl_only = set(pkl_pids.keys()) - detail_pids
    detail_only = detail_pids - set(pkl_pids.keys())
    
    print("Matching analysis:")
    print(f"  Common PIDs: {len(common)}")
    print(f"  PKL only: {len(pkl_only)}")
    print(f"  Detail only: {len(detail_only)}")
    print()
    
    # 5. 分析PKL中的patch ID是否符合公式
    print("Checking if PKL PIDs follow the formula (patchid = inst_id * max_inst + i):")
    sample_pkl_pids = sorted(list(pkl_pids.keys()))[:20]
    print("Sample PKL PIDs and their calculated inst_id/i:")
    for pid in sample_pkl_pids:
        inst_id = pid // 20
        i = pid % 20
        expected_pid = inst_id * 20 + i
        in_detail = pid in detail_pids
        print(f"  PID {pid}: inst_id={inst_id}, i={i}, expected={expected_pid}, in_detail={in_detail}")
    print()
    
    # 6. 检查detail文件中的patch ID是否符合公式
    print("Checking if detail PIDs follow the formula:")
    sample_detail_pids = sorted(list(detail_pids))[:20]
    print("Sample detail PIDs and their calculated inst_id/i:")
    for pid in sample_detail_pids:
        inst_id = pid // 20
        i = pid % 20
        expected_pid = inst_id * 20 + i
        in_pkl = pid in pkl_pids
        print(f"  PID {pid}: inst_id={inst_id}, i={i}, expected={expected_pid}, in_pkl={in_pkl}")
    print()
    
    # 7. 检查为什么PKL中只有67282个结果
    print("Analysis: Why PKL has fewer results than detail files?")
    print(f"  Detail files contain {len(detail_pids)} patch IDs")
    print(f"  PKL contains {len(pkl_pids)} patch IDs")
    print(f"  Difference: {len(detail_pids) - len(pkl_pids)}")
    print()
    print("Possible reasons:")
    print("  1. Some patches were filtered during inference")
    print("  2. Some patches failed during inference")
    print("  3. Dataset loading order differs from detail file order")
    print()
    
    # 8. 检查文件系统中的实际文件数
    patch_dir = 'tuiliBIoU_biou_0_0/patches/img_dir/val'
    if os.path.exists(patch_dir):
        actual_files = [f for f in os.listdir(patch_dir) if f.endswith('.png')]
        print(f"Actual patch files in filesystem: {len(actual_files)}")
        print(f"PKL results: {len(pkl_pids)}")
        print(f"Detail file PIDs: {len(detail_pids)}")
        print()
        print("Conclusion:")
        if len(actual_files) == len(detail_pids):
            print("  ✓ Actual files match detail file PIDs")
        if len(pkl_pids) < len(actual_files):
            print(f"  ⚠ PKL has fewer results ({len(pkl_pids)}) than actual files ({len(actual_files)})")
            print("     This suggests some patches were not processed during inference")

if __name__ == '__main__':
    analyze()

