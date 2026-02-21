#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算不同BIoU阈值推理结果的FPS（Frames Per Second）
"""
import os
import json
import argparse
import time
from pathlib import Path
from collections import defaultdict


def get_image_count_from_json(json_path):
    """从JSON文件获取图像数量"""
    if not os.path.exists(json_path):
        return 0
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            # COCO格式
            if 'images' in data:
                return len(data['images'])
            elif 'annotations' in data:
                # 统计唯一的image_id
                image_ids = set(ann['image_id'] for ann in data['annotations'])
                return len(image_ids)
        return 0
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return 0


def get_patch_count(patches_dir):
    """统计补丁数量"""
    if not os.path.exists(patches_dir):
        return 0
    
    img_dir = os.path.join(patches_dir, 'img_dir', 'val')
    if os.path.exists(img_dir):
        return len([f for f in os.listdir(img_dir) if f.endswith('.png')])
    return 0


def calculate_inference_time(result_dir):
    """
    计算推理时间
    通过检查文件修改时间估算
    """
    result_path = Path(result_dir)
    
    # 检查关键文件的时间戳
    refined_pkl = result_path / 'refined.pkl'
    refined_json = result_path / 'refined.json'
    
    if not refined_pkl.exists():
        return None
    
    # 获取文件创建和修改时间
    pkl_mtime = refined_pkl.stat().st_mtime
    
    # 尝试从日志文件读取时间（如果有）
    # 或者使用文件时间差估算
    
    # 简单估算：使用pkl文件的修改时间作为推理结束时间
    # 开始时间可以通过coarse.json或patches目录估算
    patches_dir = result_path / 'patches'
    if patches_dir.exists():
        # 找到patches目录中最早的文件时间
        patch_files = list(patches_dir.rglob('*.png'))
        if patch_files:
            earliest_time = min(f.stat().st_mtime for f in patch_files)
            inference_time = pkl_mtime - earliest_time
            return max(inference_time, 0)
    
    return None


def calculate_fps_for_threshold(result_dir, method='images'):
    """
    计算单个BIoU阈值的结果FPS
    
    Args:
        result_dir: 结果目录路径
        method: 'images' 或 'patches'
            - 'images': 基于原始图像数量计算FPS
            - 'patches': 基于补丁数量计算FPS
    
    Returns:
        dict: 包含FPS和相关信息的字典
    """
    result_path = Path(result_dir)
    
    # 获取图像/补丁数量
    if method == 'images':
        coarse_json = result_path / 'coarse.json'
        count = get_image_count_from_json(str(coarse_json))
        unit = 'images'
    else:  # patches
        patches_dir = result_path / 'patches'
        count = get_patch_count(str(patches_dir))
        unit = 'patches'
    
    if count == 0:
        return {
            'fps': 0,
            'count': 0,
            'time': None,
            'unit': unit,
            'status': 'No data found'
        }
    
    # 计算推理时间
    inference_time = calculate_inference_time(result_dir)
    
    if inference_time is None or inference_time <= 0:
        return {
            'fps': 0,
            'count': count,
            'time': None,
            'unit': unit,
            'status': 'Cannot calculate time'
        }
    
    fps = count / inference_time if inference_time > 0 else 0
    
    return {
        'fps': fps,
        'count': count,
        'time': inference_time,
        'unit': unit,
        'status': 'OK'
    }


def find_result_dirs(prefix_base):
    """查找所有BIoU阈值的结果目录"""
    prefix_base = Path(prefix_base)
    parent_dir = prefix_base.parent if prefix_base.parent != Path('.') else Path('.')
    base_name = prefix_base.name
    
    result_dirs = []
    
    if parent_dir.exists():
        for item in parent_dir.iterdir():
            if item.is_dir() and item.name.startswith(base_name + '_biou_'):
                try:
                    thresh_str = item.name.replace(base_name + '_biou_', '').replace('_', '.')
                    biou_thresh = float(thresh_str)
                    result_dirs.append((biou_thresh, item))
                except ValueError:
                    continue
    
    result_dirs.sort(key=lambda x: x[0])
    return result_dirs


def main():
    parser = argparse.ArgumentParser(
        description='Calculate FPS for BIoU threshold sweep results'
    )
    parser.add_argument(
        'prefix_base',
        help='Base path for result directories (e.g., tuiliBIoU)'
    )
    parser.add_argument(
        '--method',
        choices=['images', 'patches'],
        default='images',
        help='FPS calculation method: images (based on original images) or patches (based on patches)'
    )
    parser.add_argument(
        '--output-csv',
        default=None,
        help='Path to save results as CSV file (optional)'
    )
    parser.add_argument(
        '--single',
        default=None,
        type=float,
        help='Calculate FPS for a single BIoU threshold'
    )
    
    args = parser.parse_args()
    
    if args.single is not None:
        # 计算单个结果
        prefix_base = Path(args.prefix_base)
        result_dir = prefix_base.parent / f"{prefix_base.name}_biou_{args.single:.3f}".replace('.', '_')
        
        if not result_dir.exists():
            print(f"Error: Result directory not found: {result_dir}")
            return
        
        print(f"Calculating FPS for BIoU threshold: {args.single:.3f}")
        result = calculate_fps_for_threshold(str(result_dir), method=args.method)
        
        print(f"\nResults:")
        print(f"  Method: {args.method}")
        print(f"  Count: {result['count']} {result['unit']}")
        if result['time']:
            print(f"  Time: {result['time']:.2f} seconds")
            print(f"  FPS: {result['fps']:.4f} {result['unit']}/s")
        else:
            print(f"  Status: {result['status']}")
    else:
        # 计算所有结果
        result_dirs = find_result_dirs(args.prefix_base)
        
        if not result_dirs:
            print(f"Error: No result directories found with prefix: {args.prefix_base}")
            return
        
        print(f"\n{'='*80}")
        print(f"FPS Calculation for BIoU Threshold Sweep")
        print(f"{'='*80}")
        print(f"Method: {args.method}")
        print(f"Found {len(result_dirs)} result directories")
        print(f"{'='*80}\n")
        
        all_results = {}
        
        for biou_thresh, result_dir in result_dirs:
            print(f"Processing BIoU threshold: {biou_thresh:.3f}")
            result = calculate_fps_for_threshold(str(result_dir), method=args.method)
            all_results[biou_thresh] = result
            
            if result['time']:
                print(f"  ✓ Count: {result['count']} {result['unit']}, "
                      f"Time: {result['time']:.2f}s, "
                      f"FPS: {result['fps']:.4f} {result['unit']}/s\n")
            else:
                print(f"  ✗ {result['status']}\n")
        
        # 打印汇总表格
        print(f"\n{'='*80}")
        print(f"FPS Summary Table")
        print(f"{'='*80}")
        print(f"{'BIoU Thresh':<12} {'Count':<10} {'Time (s)':<12} {'FPS':<12} {'Status':<20}")
        print(f"{'-'*80}")
        
        for thresh in sorted(all_results.keys()):
            r = all_results[thresh]
            if r['time']:
                print(f"{thresh:<12.3f} {r['count']:<10} {r['time']:<12.2f} {r['fps']:<12.4f} {r['status']:<20}")
            else:
                print(f"{thresh:<12.3f} {r['count']:<10} {'N/A':<12} {'N/A':<12} {r['status']:<20}")
        
        print(f"{'='*80}\n")
        
        # 找出最佳FPS
        valid_results = {k: v for k, v in all_results.items() if v['time'] and v['fps'] > 0}
        if valid_results:
            best_fps = max(valid_results.items(), key=lambda x: x[1]['fps'])
            print(f"Best FPS: BIoU={best_fps[0]:.3f}, FPS={best_fps[1]['fps']:.4f} {best_fps[1]['unit']}/s\n")
        
        # 保存到CSV
        if args.output_csv:
            import csv
            with open(args.output_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['BIoU_Thresh', 'Count', 'Time_seconds', 'FPS', 'Unit', 'Status'])
                for thresh in sorted(all_results.keys()):
                    r = all_results[thresh]
                    writer.writerow([
                        f"{thresh:.3f}",
                        r['count'],
                        f"{r['time']:.6f}" if r['time'] else 'N/A',
                        f"{r['fps']:.6f}" if r['time'] else 'N/A',
                        r['unit'],
                        r['status']
                    ])
            print(f"Results saved to: {args.output_csv}")


if __name__ == '__main__':
    main()

