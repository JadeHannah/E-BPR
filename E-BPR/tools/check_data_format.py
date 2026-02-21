#!/usr/bin/env python
"""检查数据格式的脚本

使用方法:
    python tools/check_data_format.py --data-root /path/to/data

功能:
    1. 检查ground truth mask的格式（应该是0和1，不是0和255）
    2. 统计前景/背景像素比例
    3. 检查数据加载是否正确
"""

import os
import argparse
import numpy as np
from pathlib import Path
import cv2
from collections import Counter

def check_mask_format(mask_path):
    """检查单个mask文件的格式"""
    if not os.path.exists(mask_path):
        return None
    
    # 尝试不同的加载方式
    try:
        # 方式1: 作为图像加载
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            # 方式2: 作为numpy数组加载
            mask = np.load(mask_path)
    except:
        try:
            mask = np.load(mask_path)
        except:
            print(f"无法加载: {mask_path}")
            return None
    
    if mask is None:
        return None
    
    # 获取唯一值
    unique_values = np.unique(mask)
    
    return {
        'path': mask_path,
        'shape': mask.shape,
        'dtype': mask.dtype,
        'unique_values': unique_values,
        'min': mask.min(),
        'max': mask.max(),
        'fg_pixels': (mask > 0).sum() if len(unique_values) > 1 else 0,
        'bg_pixels': (mask == 0).sum(),
        'total_pixels': mask.size
    }

def check_dataset_format(data_root, dataset_type='train', num_samples=10):
    """检查数据集格式"""
    data_root = Path(data_root)
    
    # 根据RefineDataset的结构查找文件
    ann_dir = data_root / 'ann_dir' / dataset_type
    mask_dir = data_root / 'mask_dir' / dataset_type
    
    if not ann_dir.exists() and not mask_dir.exists():
        print(f"错误: 找不到标注目录 {ann_dir} 或 {mask_dir}")
        return
    
    # 优先使用ann_dir
    if ann_dir.exists():
        mask_files = list(ann_dir.glob('*.png')) + list(ann_dir.glob('*.jpg'))
        if len(mask_files) == 0:
            mask_files = list(ann_dir.glob('*.npy'))
        mask_dir_path = ann_dir
    elif mask_dir.exists():
        mask_files = list(mask_dir.glob('*.png')) + list(mask_dir.glob('*.jpg'))
        if len(mask_files) == 0:
            mask_files = list(mask_dir.glob('*.npy'))
        mask_dir_path = mask_dir
    else:
        print(f"错误: 找不到标注文件")
        return
    
    if len(mask_files) == 0:
        print(f"错误: 在 {mask_dir_path} 中找不到标注文件")
        return
    
    print(f"\n找到 {len(mask_files)} 个标注文件")
    print(f"检查前 {min(num_samples, len(mask_files))} 个样本...\n")
    
    # 检查样本
    all_unique_values = set()
    all_fg_ratios = []
    all_bg_ratios = []
    format_issues = []
    
    for i, mask_file in enumerate(mask_files[:num_samples]):
        info = check_mask_format(str(mask_file))
        if info is None:
            continue
        
        print(f"样本 {i+1}: {mask_file.name}")
        print(f"  形状: {info['shape']}")
        print(f"  数据类型: {info['dtype']}")
        print(f"  唯一值: {info['unique_values']}")
        print(f"  范围: [{info['min']}, {info['max']}]")
        
        # 检查格式问题
        if 255 in info['unique_values']:
            format_issues.append(f"{mask_file.name}: 包含255值（应该是0和1）")
            print(f"  ⚠️  警告: 包含255值！应该转换为0和1")
        
        if len(info['unique_values']) > 2:
            format_issues.append(f"{mask_file.name}: 包含多个值 {info['unique_values']}")
            print(f"  ⚠️  警告: 包含多个唯一值，二分类应该只有0和1")
        
        # 统计前景/背景比例
        if info['total_pixels'] > 0:
            fg_ratio = info['fg_pixels'] / info['total_pixels']
            bg_ratio = info['bg_pixels'] / info['total_pixels']
            all_fg_ratios.append(fg_ratio)
            all_bg_ratios.append(bg_ratio)
            print(f"  前景像素: {info['fg_pixels']} ({fg_ratio*100:.2f}%)")
            print(f"  背景像素: {info['bg_pixels']} ({bg_ratio*100:.2f}%)")
        
        all_unique_values.update(info['unique_values'])
        print()
    
    # 总结
    print("=" * 60)
    print("检查总结:")
    print("=" * 60)
    print(f"所有样本的唯一值: {sorted(all_unique_values)}")
    
    if format_issues:
        print(f"\n⚠️  发现 {len(format_issues)} 个格式问题:")
        for issue in format_issues:
            print(f"  - {issue}")
        print("\n建议: 确保mask值只有0（背景）和1（前景）")
    else:
        print("\n✅ 数据格式正确: 所有mask只包含0和1")
    
    if all_fg_ratios:
        avg_fg_ratio = np.mean(all_fg_ratios)
        avg_bg_ratio = np.mean(all_bg_ratios)
        print(f"\n平均前景比例: {avg_fg_ratio*100:.2f}%")
        print(f"平均背景比例: {avg_bg_ratio*100:.2f}%")
        
        if avg_fg_ratio < 0.1:
            print(f"\n⚠️  警告: 前景像素比例 < 10%，类别严重不平衡！")
            print("建议: 增加Dice Loss权重（dice_weight=2.0或更高）")
        elif avg_fg_ratio < 0.2:
            print(f"\n⚠️  注意: 前景像素比例 < 20%，存在类别不平衡")
            print("建议: 适当增加Dice Loss权重")
        else:
            print(f"\n✅ 类别分布相对平衡")

def check_via_dataloader(data_root, config_path):
    """通过数据加载器检查（更准确）"""
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from mmseg.datasets import build_dataset
        from mmcv import Config
        
        # 加载配置
        cfg = Config.fromfile(config_path)
        cfg.data_root = data_root
        cfg.data.train.data_root = data_root
        
        # 构建数据集
        dataset = build_dataset(cfg.data.train)
        
        print(f"\n通过数据加载器检查...")
        print(f"数据集大小: {len(dataset)}")
        
        # 检查几个样本
        for i in range(min(5, len(dataset))):
            result = dataset[i]
            if 'gt_semantic_seg' in result:
                mask = result['gt_semantic_seg']
                if hasattr(mask, 'numpy'):
                    mask = mask.numpy()
                elif hasattr(mask, 'data'):
                    mask = mask.data.numpy()
                
                unique_values = np.unique(mask)
                print(f"\n样本 {i+1} (通过数据加载器):")
                print(f"  Mask形状: {mask.shape}")
                print(f"  Mask数据类型: {mask.dtype}")
                print(f"  唯一值: {unique_values}")
                print(f"  范围: [{mask.min()}, {mask.max()}]")
                
                if 255 in unique_values:
                    print(f"  ⚠️  警告: 包含255值！")
                if len(unique_values) > 2:
                    print(f"  ⚠️  警告: 包含多个值")
    except Exception as e:
        print(f"\n无法通过数据加载器检查: {e}")
        print("将使用直接文件检查方式...")

def main():
    parser = argparse.ArgumentParser(description='检查数据格式')
    parser.add_argument('--data-root', type=str, required=True,
                        help='数据根目录路径')
    parser.add_argument('--dataset-type', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='检查哪个数据集 (train/val/test)')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='检查的样本数量')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径（用于数据加载器检查）')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("数据格式检查工具")
    print("=" * 60)
    print(f"数据根目录: {args.data_root}")
    print(f"数据集类型: {args.dataset_type}")
    print(f"检查样本数: {args.num_samples}")
    
    # 方式1: 直接检查文件
    check_dataset_format(args.data_root, args.dataset_type, args.num_samples)
    
    # 方式2: 通过数据加载器检查（如果提供了配置文件）
    if args.config and os.path.exists(args.config):
        check_via_dataloader(args.data_root, args.config)

if __name__ == '__main__':
    main()

