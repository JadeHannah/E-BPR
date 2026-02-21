#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查PKL中的patch ID和detail文件中的patch ID是否匹配
"""
import os
import pickle as pkl
import sys

def check_patch_id_mapping(pkl_file, detail_dir):
    """检查PKL中的patch ID和detail文件中的patch ID是否匹配"""
    
    # 1. 从PKL文件中提取所有patch ID
    print("Step 1: Loading patch IDs from PKL file...")
    with open(pkl_file, 'rb') as f:
        _res = pkl.load(f)
    img_infos, masks = _res
    
    pkl_pids = set()
    for img_info in img_infos:
        if 'filename' in img_info:
            filename = img_info['filename']
            try:
                pid = int(filename.replace('.png', ''))
                pkl_pids.add(pid)
            except ValueError:
                pass
    
    print(f"Total PIDs in PKL: {len(pkl_pids)}")
    if pkl_pids:
        print(f"PKL PID range: {min(pkl_pids)} - {max(pkl_pids)}")
    print()
    
    # 2. 从detail文件中提取所有patch ID
    print("Step 2: Loading patch IDs from detail files...")
    detail_files = [f for f in os.listdir(detail_dir) if f.endswith('.txt')]
    print(f"Total detail files: {len(detail_files)}")
    
    detail_pids = set()
    detail_pid_to_inst = {}  # pid -> inst_id
    
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
    
    print(f"Total PIDs in detail files: {len(detail_pids)}")
    if detail_pids:
        print(f"Detail PID range: {min(detail_pids)} - {max(detail_pids)}")
    print()
    
    # 3. 比较
    print("Step 3: Comparing PIDs...")
    pkl_only = pkl_pids - detail_pids
    detail_only = detail_pids - pkl_pids
    common = pkl_pids & detail_pids
    
    print(f"PIDs in PKL only: {len(pkl_only)}")
    print(f"PIDs in detail only: {len(detail_only)}")
    print(f"Common PIDs: {len(common)}")
    print()
    
    # 4. 分析不匹配的原因
    if pkl_only:
        print("Sample PIDs in PKL but not in detail (first 20):")
        sample_pkl_only = sorted(list(pkl_only))[:20]
        for pid in sample_pkl_only:
            print(f"  {pid}")
        print()
    
    if detail_only:
        print("Sample PIDs in detail but not in PKL (first 20):")
        sample_detail_only = sorted(list(detail_only))[:20]
        for pid in sample_detail_only:
            inst_id = detail_pid_to_inst.get(pid, 'unknown')
            print(f"  {pid} (from inst_id {inst_id})")
        print()
    
    # 5. 检查patch ID的计算公式
    print("Step 4: Checking patch ID formula...")
    print("Formula: patchid = inst_id * max_inst + i")
    print("Checking if detail PIDs match this formula...")
    
    # 从detail文件中提取一些样本
    sample_detail_file = detail_files[0] if detail_files else None
    if sample_detail_file:
        try:
            inst_id = int(sample_detail_file.replace('.txt', ''))
            detail_path = os.path.join(detail_dir, sample_detail_file)
            with open(detail_path, 'r') as f:
                lines = f.readlines()
            
            if lines:
                first_pid = int(lines[0].split()[0])
                print(f"\nSample detail file: {sample_detail_file} (inst_id={inst_id})")
                print(f"First PID: {first_pid}")
                print(f"Expected PID range for inst_id {inst_id}: {inst_id * 20} - {inst_id * 20 + 19}")
                print(f"Does first PID match? {inst_id * 20 <= first_pid <= inst_id * 20 + 19}")
        except Exception as e:
            print(f"Error checking formula: {e}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python check_patch_id_mapping.py <pkl_file> <detail_dir>")
        print("Example: python check_patch_id_mapping.py tuiliBIoU_biou_0_0/refined.pkl tuiliBIoU_biou_0_0/patches/detail_dir/val")
        sys.exit(1)
    
    pkl_file = sys.argv[1]
    detail_dir = sys.argv[2]
    check_patch_id_mapping(pkl_file, detail_dir)

