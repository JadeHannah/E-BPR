#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
绘制IoU vs BIoU散点趋势图
横轴：原始IoU值
纵轴：BIoU值
显示趋势变化，包括散点、均值±标准差、y=x参考线
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import os.path as osp
from pathlib import Path
import json
from tqdm import tqdm
from pycocotools.coco import COCO
from PIL import Image
import cv2

# 导入BIoU计算函数
import sys
sys.path.insert(0, osp.dirname(__file__))
from split_patches import compute_biou, cal_iou


def load_patch_data(patches_dir, gt_json, mode='val'):
    """
    加载所有补丁的IoU和BIoU数据
    
    Args:
        patches_dir: 补丁目录路径
        gt_json: GT JSON文件路径
        mode: 'val' 或 'train'
    
    Returns:
        iou_list: IoU值列表
        biou_list: BIoU值列表
    """
    coco = COCO(gt_json)
    
    # 补丁目录结构
    img_dir = osp.join(patches_dir, 'img_dir', mode)
    mask_dir = osp.join(patches_dir, 'mask_dir', mode)
    ann_dir = osp.join(patches_dir, 'ann_dir', mode)
    detail_dir = osp.join(patches_dir, 'detail_dir', mode)
    
    if not osp.exists(img_dir):
        raise ValueError(f"Patches directory not found: {img_dir}")
    
    # 获取所有补丁文件
    patch_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    
    print(f"Found {len(patch_files)} patches")
    
    iou_list = []
    biou_list = []
    
    # BIoU参数（与推理时一致）
    biou_dilation_ratio = 0.005
    small_patch_dilation = 5
    
    for patch_file in tqdm(patch_files, desc="Computing IoU and BIoU"):
        patch_id = patch_file.replace('.png', '')
        
        # 加载补丁
        dt_patch_path = osp.join(mask_dir, patch_file)
        gt_patch_path = osp.join(ann_dir, patch_file)
        
        if not osp.exists(dt_patch_path) or not osp.exists(gt_patch_path):
            continue
        
        # 读取mask
        dt_mask = np.array(Image.open(dt_patch_path))
        gt_mask = np.array(Image.open(gt_patch_path))
        
        # 确保是二值mask
        if dt_mask.ndim == 3:
            dt_mask = dt_mask[:, :, 0]
        if gt_mask.ndim == 3:
            gt_mask = gt_mask[:, :, 0]
        
        dt_mask = (dt_mask > 0).astype(np.uint8)
        gt_mask = (gt_mask > 0).astype(np.uint8)
        
        # 计算IoU
        iou = cal_iou(dt_mask, gt_mask)
        
        # 计算BIoU
        final_iou, iou_biou, biou = compute_biou(
            dt_mask, gt_mask,
            dilation_ratio=biou_dilation_ratio,
            combine_with_iou=False,
            small_patch_dilation=small_patch_dilation
        )
        
        iou_list.append(iou)
        biou_list.append(biou)
    
    return np.array(iou_list), np.array(biou_list)


def compute_binned_stats(iou_values, biou_values, num_bins=20):
    """
    按IoU区间分组，计算每组的均值±标准差
    
    Args:
        iou_values: IoU值数组
        biou_values: BIoU值数组
        num_bins: 分组数量
    
    Returns:
        bin_centers: 区间中心点
        bin_means: 每组的BIoU均值
        bin_stds: 每组的BIoU标准差
    """
    # 创建区间
    bins = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(iou_values, bins) - 1
    
    # 确保索引在有效范围内
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)
    
    bin_centers = []
    bin_means = []
    bin_stds = []
    
    for i in range(num_bins):
        mask = (bin_indices == i)
        if mask.sum() > 0:
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            bin_means.append(biou_values[mask].mean())
            bin_stds.append(biou_values[mask].std())
        else:
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            bin_means.append(np.nan)
            bin_stds.append(np.nan)
    
    return np.array(bin_centers), np.array(bin_means), np.array(bin_stds)


def plot_iou_vs_biou(iou_values, biou_values, output_path, title="IoU vs BIoU"):
    """
    绘制IoU vs BIoU散点趋势图
    
    Args:
        iou_values: IoU值数组
        biou_values: BIoU值数组
        output_path: 输出图片路径
        title: 图表标题
    """
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 绘制y=x参考线
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='y = x', alpha=0.7)
    
    # 绘制散点图（使用半透明，避免过度重叠）
    ax.scatter(iou_values, biou_values, 
               c='pink', s=1, alpha=0.3, label='Instance', edgecolors='none')
    
    # 计算分组统计
    bin_centers, bin_means, bin_stds = compute_binned_stats(iou_values, biou_values, num_bins=20)
    
    # 过滤NaN值
    valid_mask = ~np.isnan(bin_means)
    bin_centers = bin_centers[valid_mask]
    bin_means = bin_means[valid_mask]
    bin_stds = bin_stds[valid_mask]
    
    # 绘制均值±标准差
    ax.errorbar(bin_centers, bin_means, yerr=bin_stds,
                fmt='o', color='blue', markersize=6, 
                capsize=3, capthick=1.5, elinewidth=1.5,
                label='Mean and Std', alpha=0.8)
    
    # 设置坐标轴
    ax.set_xlabel('IoU (before refinement)', fontsize=12)
    ax.set_ylabel('BIoU (after refinement)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect('equal')
    
    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#f5f5f5')
    
    # 添加图例
    ax.legend(loc='lower right', fontsize=10)
    
    # 添加统计信息
    correlation = np.corrcoef(iou_values, biou_values)[0, 1]
    mean_improvement = np.mean(biou_values - iou_values)
    
    stats_text = f'Correlation: {correlation:.3f}\nMean Improvement: {mean_improvement:.3f}'
    ax.text(0.02, 0.98, stats_text, 
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize IoU vs BIoU scatter plot')
    parser.add_argument('patches_dir', help='Path to patches directory')
    parser.add_argument('gt_json', help='Path to GT JSON file')
    parser.add_argument('--mode', default='val', choices=['train', 'val'],
                       help='Dataset mode (default: val)')
    parser.add_argument('--output', default='iou_vs_biou.png',
                       help='Output image path (default: iou_vs_biou.png)')
    parser.add_argument('--title', default='IoU vs BIoU',
                       help='Plot title (default: IoU vs BIoU)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("IoU vs BIoU Visualization")
    print("="*80)
    print(f"Patches directory: {args.patches_dir}")
    print(f"GT JSON: {args.gt_json}")
    print(f"Mode: {args.mode}")
    print(f"Output: {args.output}")
    print("="*80)
    print()
    
    # 加载数据
    print("Loading patch data...")
    iou_values, biou_values = load_patch_data(args.patches_dir, args.gt_json, args.mode)
    
    print(f"\nLoaded {len(iou_values)} patches")
    print(f"IoU range: [{iou_values.min():.3f}, {iou_values.max():.3f}]")
    print(f"BIoU range: [{biou_values.min():.3f}, {biou_values.max():.3f}]")
    print(f"Mean IoU: {iou_values.mean():.3f}")
    print(f"Mean BIoU: {biou_values.mean():.3f}")
    print(f"Correlation: {np.corrcoef(iou_values, biou_values)[0, 1]:.3f}")
    
    # 绘制图表
    print("\nPlotting...")
    plot_iou_vs_biou(iou_values, biou_values, args.output, args.title)
    
    print("\nDone!")


if __name__ == '__main__':
    main()

