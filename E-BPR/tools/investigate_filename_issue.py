#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
调查filename字段的问题
"""
import sys
import os
sys.path.insert(0, '.')

from mmseg.datasets import build_dataset
import mmcv

def investigate():
    cfg = mmcv.Config.fromfile('configs/bpr/hrnet18s_128.py')
    cfg.data.test.data_root = 'tuiliBIoU_biou_0_0/patches'
    
    print("Building dataset...")
    dataset = build_dataset(cfg.data.test)
    
    print(f"Total samples: {len(dataset.img_infos)}")
    print(f"First img_info keys: {list(dataset.img_infos[0].keys())}")
    print(f"First img_info: {dataset.img_infos[0]}")
    print()
    
    # 检查是否有id字段
    print("Checking for 'id' field:")
    has_id = any('id' in info for info in dataset.img_infos[:100])
    print(f"Has id field: {has_id}")
    if has_id:
        sample_ids = [info.get('id') for info in dataset.img_infos[:5] if 'id' in info]
        print(f"Sample id values: {sample_ids}")
    print()
    
    # 检查filename和实际文件的关系
    print("Checking filename vs actual files:")
    patch_dir = 'tuiliBIoU_biou_0_0/patches/img_dir/val'
    actual_files = set([f for f in os.listdir(patch_dir) if f.endswith('.png')])
    print(f"Total actual files: {len(actual_files)}")
    
    dataset_filenames = set([info['filename'] for info in dataset.img_infos])
    print(f"Total dataset filenames: {len(dataset_filenames)}")
    
    matched = dataset_filenames & actual_files
    print(f"Matched filenames: {len(matched)}")
    print(f"Dataset only: {len(dataset_filenames - actual_files)}")
    print(f"Actual only: {len(actual_files - dataset_filenames)}")
    print()
    
    # 检查前10个filename
    print("First 10 filenames in dataset:")
    for i, info in enumerate(dataset.img_infos[:10]):
        filename = info['filename']
        exists = filename in actual_files
        print(f"  {i}: {filename} (exists: {exists})")

if __name__ == '__main__':
    investigate()

