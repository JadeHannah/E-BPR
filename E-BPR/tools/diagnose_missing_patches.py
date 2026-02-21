#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断missing patch mask问题
检查detail文件中的patch ID和推理结果中的patch ID是否匹配
"""
import os
import os.path as osp
import pickle as pkl
import argparse
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description='Diagnose missing patch masks')
    parser.add_argument('result_dir', help='Result directory (e.g., tuiliBIoU_biou_0_0)')
    parser.add_argument('--mode', default='val', choices=['train', 'val'],
                       help='Dataset mode (default: val)')
    
    args = parser.parse_args()
    
    detail_dir = osp.join(args.result_dir, 'patches', 'detail_dir', args.mode)
    pkl_file = osp.join(args.result_dir, 'refined.pkl')
    
    if not osp.exists(detail_dir):
        print(f"Error: Detail directory not found: {detail_dir}")
        return
    
    if not osp.exists(pkl_file):
        print(f"Error: PKL file not found: {pkl_file}")
        return
    
    print("="*80)
    print("Diagnosing Missing Patch Masks")
    print("="*80)
    print(f"Detail dir: {detail_dir}")
    print(f"PKL file: {pkl_file}")
    print("="*80)
    print()
    
    # 1. 从detail文件中提取所有patch ID
    print("Step 1: Loading patch IDs from detail files...")
    detail_pids = set()
    inst_pid_map = defaultdict(list)  # inst_id -> [pid1, pid2, ...]
    
    detail_files = sorted([f for f in os.listdir(detail_dir) if f.endswith('.txt')])
    print(f"Found {len(detail_files)} detail files")
    
    for fname in detail_files:
        inst_id = int(fname.replace('.txt', ''))
        fpath = osp.join(detail_dir, fname)
        with open(fpath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if parts:
                    try:
                        pid = int(parts[0])
                        detail_pids.add(pid)
                        inst_pid_map[inst_id].append(pid)
                    except ValueError:
                        continue
    
    print(f"Total patch IDs in detail files: {len(detail_pids)}")
    print(f"Detail PID range: {min(detail_pids)} - {max(detail_pids)}")
    print()
    
    # 2. 从PKL文件中提取所有patch ID
    print("Step 2: Loading patch IDs from PKL file...")
    with open(pkl_file, 'rb') as f:
        _res = pkl.load(f)  # img_infos, masks
    
    img_infos, masks = _res
    print(f"Total items in PKL: {len(img_infos)}")
    
    pkl_pids = set()
    pkl_pid_to_info = {}  # pid -> img_info
    
    for img_info, mask in zip(img_infos, masks):
        try:
            # 方法1：从filename提取（优先，因为filename就是patch ID）
            if 'filename' in img_info:
                filename = img_info['filename']
                pid = int(filename.replace('.png', ''))
            # 方法2：从seg_map提取（备用）
            elif 'ann' in img_info and 'seg_map' in img_info['ann']:
                seg_map = img_info['ann']['seg_map']
                pid = int(seg_map.split('.')[0])
            else:
                raise KeyError("Neither 'filename' nor 'ann.seg_map' found")
            
            pkl_pids.add(pid)
            pkl_pid_to_info[pid] = img_info
        except (KeyError, ValueError, AttributeError) as e:
            print(f"Warning: Could not extract PID from img_info: {e}")
            print(f"  img_info keys: {list(img_info.keys())}")
            if 'ann' in img_info:
                print(f"  ann keys: {list(img_info['ann'].keys())}")
            if 'filename' in img_info:
                print(f"  filename value: {img_info['filename']}")
            continue
    
    print(f"Total patch IDs in PKL: {len(pkl_pids)}")
    if pkl_pids:
        print(f"PKL PID range: {min(pkl_pids)} - {max(pkl_pids)}")
    print()
    
    # 3. 找出缺失的补丁
    print("Step 3: Finding missing patches...")
    missing_pids = detail_pids - pkl_pids
    extra_pids = pkl_pids - detail_pids
    
    print(f"Missing patches (in detail but not in PKL): {len(missing_pids)}")
    print(f"Extra patches (in PKL but not in detail): {len(extra_pids)}")
    print()
    
    if missing_pids:
        print(f"First 20 missing PIDs: {sorted(list(missing_pids))[:20]}")
        print()
        
        # 按inst_id分组统计
        missing_by_inst = defaultdict(int)
        for inst_id, pids in inst_pid_map.items():
            missing_count = sum(1 for pid in pids if pid in missing_pids)
            if missing_count > 0:
                missing_by_inst[inst_id] = missing_count
        
        print(f"Instances with missing patches: {len(missing_by_inst)}")
        if missing_by_inst:
            print("Top 10 instances with most missing patches:")
            for inst_id, count in sorted(missing_by_inst.items(), key=lambda x: x[1], reverse=True)[:10]:
                total = len(inst_pid_map[inst_id])
                print(f"  inst_id {inst_id}: {count}/{total} patches missing")
        print()
    
    # 4. 分析原因
    print("Step 4: Analysis...")
    if len(missing_pids) == 0:
        print("✓ No missing patches! All patches are present in PKL.")
    else:
        missing_ratio = len(missing_pids) / len(detail_pids) * 100
        print(f"⚠️  {missing_ratio:.2f}% of patches are missing from PKL")
        print()
        print("Possible reasons:")
        print("1. Patches were filtered out during inference (but detail files still record them)")
        print("2. Inference failed for some patches")
        print("3. Patch ID mismatch between detail files and PKL")
        print()
        print("Impact:")
        if missing_ratio < 5:
            print("  Low impact: Most patches are available, fallback to coarse mask is acceptable")
        elif missing_ratio < 20:
            print("  Medium impact: Some patches missing, may affect quality")
        else:
            print("  High impact: Many patches missing, may significantly affect results")
    
    print()
    print("="*80)
    print("Diagnosis complete!")
    print("="*80)


if __name__ == '__main__':
    main()

