#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量生成所有BIoU阈值的IoU vs BIoU散点图
"""
import os
import subprocess
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Generate IoU vs BIoU plots for all thresholds')
    parser.add_argument('--prefix-base', default='tuiliBIoU',
                       help='Base prefix for result directories (default: tuiliBIoU)')
    parser.add_argument('--gt-json', default='data/cityscapes/annotations/instancesonly_filtered_gtFine_val.json',
                       help='Path to GT JSON file')
    parser.add_argument('--output-dir', default='visualizations',
                       help='Output directory for plots (default: visualizations)')
    parser.add_argument('--mode', default='val', choices=['train', 'val'],
                       help='Dataset mode (default: val)')
    parser.add_argument('--thresholds', nargs='+', type=float,
                       default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
                       help='List of BIoU thresholds to visualize')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("Generating IoU vs BIoU plots for all thresholds")
    print("="*80)
    print(f"Prefix base: {args.prefix_base}")
    print(f"GT JSON: {args.gt_json}")
    print(f"Output directory: {args.output_dir}")
    print(f"Thresholds: {args.thresholds}")
    print("="*80)
    print()
    
    script_path = Path(__file__).parent / 'visualize_iou_vs_biou.py'
    
    for thresh in args.thresholds:
        # 构建目录名
        thresh_str = f"{thresh:.3f}".replace('.', '_')
        patches_dir = f"{args.prefix_base}_biou_{thresh_str}/patches"
        
        # 检查目录是否存在
        if not os.path.exists(patches_dir):
            print(f"⚠️  Skipping threshold {thresh}: patches directory not found: {patches_dir}")
            continue
        
        # 构建输出文件名
        output_file = os.path.join(args.output_dir, f"iou_vs_biou_thresh_{thresh_str}.png")
        
        print(f"Processing threshold {thresh}...")
        print(f"  Patches dir: {patches_dir}")
        print(f"  Output: {output_file}")
        
        # 运行可视化脚本
        cmd = [
            'python', str(script_path),
            patches_dir,
            args.gt_json,
            '--mode', args.mode,
            '--output', output_file,
            '--title', f'IoU vs BIoU (threshold={thresh})'
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"  ✓ Generated: {output_file}\n")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Error generating plot: {e}\n")
    
    print("="*80)
    print(f"All plots saved to: {args.output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()

