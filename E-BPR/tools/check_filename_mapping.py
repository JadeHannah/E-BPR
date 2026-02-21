#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查PKL文件中的filename和实际文件系统中的文件名是否匹配
"""
import os
import pickle as pkl
import sys

def check_filename_mapping(pkl_file, patch_dir):
    """检查PKL中的filename和实际文件的映射关系"""
    
    # 加载PKL文件
    with open(pkl_file, 'rb') as f:
        _res = pkl.load(f)
    img_infos, masks = _res
    
    print(f"Total items in PKL: {len(img_infos)}")
    print(f"Patch directory: {patch_dir}")
    print()
    
    # 检查前20个
    print("Checking first 20 img_infos:")
    print("-" * 80)
    matched = 0
    not_matched = 0
    
    for i, info in enumerate(img_infos[:20]):
        filename = info.get('filename', 'N/A')
        if filename == 'N/A':
            print(f"  {i}: No filename field")
            continue
        
        # 提取PID
        try:
            pid = int(filename.replace('.png', ''))
        except ValueError:
            print(f"  {i}: filename={filename}, cannot extract PID")
            continue
        
        # 检查文件是否存在
        actual_file = os.path.join(patch_dir, f"{pid}.png")
        exists = os.path.exists(actual_file)
        
        if exists:
            matched += 1
            status = "✓"
        else:
            not_matched += 1
            status = "✗"
        
        print(f"  {i}: {status} filename={filename}, pid={pid}, exists={exists}")
    
    print("-" * 80)
    print(f"Matched: {matched}, Not matched: {not_matched}")
    print()
    
    # 检查所有文件
    print("Checking all files...")
    all_pids_from_pkl = set()
    for info in img_infos:
        filename = info.get('filename', 'N/A')
        if filename != 'N/A':
            try:
                pid = int(filename.replace('.png', ''))
                all_pids_from_pkl.add(pid)
            except ValueError:
                pass
    
    # 获取实际文件系统中的所有PID
    if os.path.exists(patch_dir):
        actual_files = [f for f in os.listdir(patch_dir) if f.endswith('.png')]
        all_pids_from_files = set()
        for f in actual_files:
            try:
                pid = int(f.replace('.png', ''))
                all_pids_from_files.add(pid)
            except ValueError:
                pass
        
        print(f"PIDs in PKL: {len(all_pids_from_pkl)}")
        print(f"PIDs in filesystem: {len(all_pids_from_files)}")
        print(f"PIDs in PKL but not in files: {len(all_pids_from_pkl - all_pids_from_files)}")
        print(f"PIDs in files but not in PKL: {len(all_pids_from_files - all_pids_from_pkl)}")
        
        # 显示一些不匹配的示例
        missing_in_files = list(all_pids_from_pkl - all_pids_from_files)[:10]
        if missing_in_files:
            print(f"\nFirst 10 PIDs in PKL but not in files: {missing_in_files}")
        
        extra_in_files = list(all_pids_from_files - all_pids_from_pkl)[:10]
        if extra_in_files:
            print(f"First 10 PIDs in files but not in PKL: {extra_in_files}")
    else:
        print(f"Patch directory does not exist: {patch_dir}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python check_filename_mapping.py <pkl_file> <patch_dir>")
        print("Example: python check_filename_mapping.py tuiliBIoU_biou_0_0/refined.pkl tuiliBIoU_biou_0_0/patches/img_dir/val")
        sys.exit(1)
    
    pkl_file = sys.argv[1]
    patch_dir = sys.argv[2]
    check_filename_mapping(pkl_file, patch_dir)

