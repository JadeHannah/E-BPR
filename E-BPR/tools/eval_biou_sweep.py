#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BIoU阈值消融实验评估脚本
批量评估不同BIoU阈值的结果，计算COCO格式的mAP等指标
"""
import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    print("Error: pycocotools not installed. Please install it with: pip install pycocotools")
    sys.exit(1)


def evaluate_single_result(pred_json, gt_json, verbose=False):
    """
    评估单个结果文件
    
    Args:
        pred_json: 预测结果JSON文件路径
        gt_json: GT JSON文件路径
        verbose: 是否打印详细信息
    
    Returns:
        dict: 包含各种评估指标的字典
    """
    if not os.path.exists(pred_json):
        print(f"Warning: Prediction file not found: {pred_json}")
        return None
    
    if not os.path.exists(gt_json):
        print(f"Error: GT file not found: {gt_json}")
        return None
    
    # 加载GT和预测结果
    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(pred_json)
    
    # 创建评估器
    coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
    
    # 运行评估
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # 提取指标
    metrics = {
        'AP': coco_eval.stats[0],      # AP @ IoU=0.50:0.95
        'AP50': coco_eval.stats[1],     # AP @ IoU=0.50
        'AP75': coco_eval.stats[2],     # AP @ IoU=0.75
        'APs': coco_eval.stats[3],      # AP for small objects
        'APm': coco_eval.stats[4],      # AP for medium objects
        'APl': coco_eval.stats[5],      # AP for large objects
        'AR1': coco_eval.stats[6],      # AR given 1 detection per image
        'AR10': coco_eval.stats[7],     # AR given 10 detections per image
        'AR100': coco_eval.stats[8],    # AR given 100 detections per image
        'ARs': coco_eval.stats[9],      # AR for small objects
        'ARm': coco_eval.stats[10],     # AR for medium objects
        'ARl': coco_eval.stats[11],     # AR for large objects
    }
    
    if verbose:
        print(f"\nDetailed metrics for {os.path.basename(pred_json)}:")
        print(f"  AP (IoU=0.50:0.95): {metrics['AP']:.4f}")
        print(f"  AP50: {metrics['AP50']:.4f}")
        print(f"  AP75: {metrics['AP75']:.4f}")
        print(f"  APs/APm/APl: {metrics['APs']:.4f}/{metrics['APm']:.4f}/{metrics['APl']:.4f}")
        print(f"  AR1/AR10/AR100: {metrics['AR1']:.4f}/{metrics['AR10']:.4f}/{metrics['AR100']:.4f}")
    
    return metrics


def find_result_dirs(prefix_base):
    """
    查找所有BIoU阈值的结果目录
    
    Args:
        prefix_base: 结果目录基础路径
    
    Returns:
        list: [(biou_thresh, result_dir), ...]
    """
    prefix_base = Path(prefix_base)
    parent_dir = prefix_base.parent if prefix_base.parent != Path('.') else Path('.')
    base_name = prefix_base.name
    
    result_dirs = []
    
    # 查找所有匹配的目录
    if parent_dir.exists():
        for item in parent_dir.iterdir():
            if item.is_dir() and item.name.startswith(base_name + '_biou_'):
                # 提取阈值
                try:
                    thresh_str = item.name.replace(base_name + '_biou_', '').replace('_', '.')
                    biou_thresh = float(thresh_str)
                    result_dirs.append((biou_thresh, item))
                except ValueError:
                    continue
    
    # 按阈值排序
    result_dirs.sort(key=lambda x: x[0])
    return result_dirs


def evaluate_sweep_results(prefix_base, gt_json, output_file=None, verbose=False):
    """
    评估所有BIoU阈值的结果
    
    Args:
        prefix_base: 结果目录基础路径
        gt_json: GT JSON文件路径
        output_file: 输出CSV文件路径（可选）
        verbose: 是否打印详细信息
    
    Returns:
        dict: {biou_thresh: metrics_dict, ...}
    """
    result_dirs = find_result_dirs(prefix_base)
    
    if not result_dirs:
        print(f"Error: No result directories found with prefix: {prefix_base}")
        return None
    
    print(f"\n{'='*80}")
    print(f"Evaluating BIoU Sweep Results")
    print(f"{'='*80}")
    print(f"Found {len(result_dirs)} result directories")
    print(f"GT JSON: {gt_json}")
    print(f"{'='*80}\n")
    
    all_results = {}
    
    for biou_thresh, result_dir in result_dirs:
        pred_json = result_dir / 'refined.json'
        
        print(f"Evaluating BIoU threshold: {biou_thresh:.3f}")
        print(f"  Result dir: {result_dir}")
        print(f"  Prediction JSON: {pred_json}")
        
        metrics = evaluate_single_result(str(pred_json), gt_json, verbose=verbose)
        
        if metrics is not None:
            all_results[biou_thresh] = metrics
            print(f"  ✓ AP: {metrics['AP']:.4f}, AP50: {metrics['AP50']:.4f}, AP75: {metrics['AP75']:.4f}\n")
        else:
            print(f"  ✗ Failed to evaluate\n")
    
    # 打印汇总表格
    print_summary_table(all_results)
    
    # 保存结果到CSV
    if output_file:
        save_results_to_csv(all_results, output_file)
        print(f"\nResults saved to: {output_file}")
    
    return all_results


def print_summary_table(results):
    """
    打印汇总表格
    
    Args:
        results: {biou_thresh: metrics_dict, ...}
    """
    if not results:
        return
    
    print(f"\n{'='*80}")
    print(f"Summary Table")
    print(f"{'='*80}")
    print(f"{'BIoU Thresh':<12} {'AP':<8} {'AP50':<8} {'AP75':<8} {'APs':<8} {'APm':<8} {'APl':<8}")
    print(f"{'-'*80}")
    
    for thresh in sorted(results.keys()):
        m = results[thresh]
        print(f"{thresh:<12.3f} {m['AP']:<8.4f} {m['AP50']:<8.4f} {m['AP75']:<8.4f} "
              f"{m['APs']:<8.4f} {m['APm']:<8.4f} {m['APl']:<8.4f}")
    
    print(f"{'='*80}\n")
    
    # 找出最佳结果
    if results:
        best_ap = max(results.items(), key=lambda x: x[1]['AP'])
        best_ap50 = max(results.items(), key=lambda x: x[1]['AP50'])
        
        print(f"Best AP: BIoU={best_ap[0]:.3f}, AP={best_ap[1]['AP']:.4f}")
        print(f"Best AP50: BIoU={best_ap50[0]:.3f}, AP50={best_ap50[1]['AP50']:.4f}\n")


def save_results_to_csv(results, output_file):
    """
    保存结果到CSV文件
    
    Args:
        results: {biou_thresh: metrics_dict, ...}
        output_file: 输出文件路径
    """
    import csv
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # 写入表头
        header = ['BIoU_Thresh', 'AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 
                  'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl']
        writer.writerow(header)
        
        # 写入数据
        for thresh in sorted(results.keys()):
            m = results[thresh]
            row = [
                f"{thresh:.3f}",
                f"{m['AP']:.6f}",
                f"{m['AP50']:.6f}",
                f"{m['AP75']:.6f}",
                f"{m['APs']:.6f}",
                f"{m['APm']:.6f}",
                f"{m['APl']:.6f}",
                f"{m['AR1']:.6f}",
                f"{m['AR10']:.6f}",
                f"{m['AR100']:.6f}",
                f"{m['ARs']:.6f}",
                f"{m['ARm']:.6f}",
                f"{m['ARl']:.6f}",
            ]
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate BIoU threshold sweep results'
    )
    parser.add_argument(
        'prefix_base',
        help='Base path for result directories (e.g., maskrcnn_val_refined)'
    )
    parser.add_argument(
        'gt_json',
        help='Path to GT JSON file'
    )
    parser.add_argument(
        '--output-csv',
        default=None,
        help='Path to save results as CSV file (optional)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed metrics for each result'
    )
    parser.add_argument(
        '--single',
        default=None,
        type=float,
        help='Evaluate only a single BIoU threshold (e.g., 0.55)'
    )
    
    args = parser.parse_args()
    
    if args.single is not None:
        # 评估单个结果
        prefix_base = Path(args.prefix_base)
        # 尝试多种格式的目录名
        thresh_str = str(args.single).replace('.', '_')
        possible_dirs = [
            prefix_base.parent / f"{prefix_base.name}_biou_{thresh_str}",  # 0_0
            prefix_base.parent / f"{prefix_base.name}_biou_{args.single:.3f}".replace('.', '_'),  # 0_000
            prefix_base.parent / f"{prefix_base.name}_biou_{args.single:.2f}".replace('.', '_'),  # 0_00
        ]
        
        result_dir = None
        for possible_dir in possible_dirs:
            pred_json = possible_dir / 'refined.json'
            if pred_json.exists():
                result_dir = possible_dir
                break
        
        if result_dir is None:
            print(f"Error: Result file not found. Tried:")
            for d in possible_dirs:
                print(f"  {d / 'refined.json'}")
            sys.exit(1)
        
        pred_json = result_dir / 'refined.json'
        
        print(f"Evaluating single result: BIoU threshold = {args.single:.3f}")
        metrics = evaluate_single_result(str(pred_json), args.gt_json, verbose=True)
        
        if metrics:
            print(f"\nResults:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")
    else:
        # 评估所有结果
        results = evaluate_sweep_results(
            args.prefix_base,
            args.gt_json,
            output_file=args.output_csv,
            verbose=args.verbose
        )
        
        if results is None:
            sys.exit(1)


if __name__ == '__main__':
    main()

