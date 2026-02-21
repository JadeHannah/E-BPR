#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估单个推理结果
"""
import sys
import os
import json
from pathlib import Path

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    print("Error: pycocotools not installed. Please install it with: pip install pycocotools")
    sys.exit(1)


def evaluate_single(pred_json, gt_json, verbose=True):
    """
    评估单个结果文件
    
    Args:
        pred_json: 预测结果JSON文件路径
        gt_json: GT JSON文件路径
        verbose: 是否打印详细信息
    """
    if not os.path.exists(pred_json):
        print(f"Error: Prediction file not found: {pred_json}")
        return None
    
    if not os.path.exists(gt_json):
        print(f"Error: GT file not found: {gt_json}")
        return None
    
    print(f"Loading GT from: {gt_json}")
    coco_gt = COCO(gt_json)
    
    print(f"Loading predictions from: {pred_json}")
    # loadRes 需要的是 annotations 列表，如果是完整的 COCO JSON，需要提取 annotations
    with open(pred_json, 'r') as f:
        pred_data = json.load(f)
    
    # 如果 pred_data 是完整的 COCO 格式，提取 annotations
    if isinstance(pred_data, dict) and 'annotations' in pred_data:
        pred_annotations = pred_data['annotations']
    elif isinstance(pred_data, list):
        pred_annotations = pred_data
    else:
        print(f"Error: Unexpected format in {pred_json}")
        return None
    
    print(f"  Found {len(pred_annotations)} annotations")
    coco_dt = coco_gt.loadRes(pred_annotations)
    
    print("Running evaluation...")
    coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    if verbose:
        print(f"\n{'='*60}")
        print("Detailed Metrics:")
        print(f"{'='*60}")
        print(f"AP @ IoU=0.50:0.95: {coco_eval.stats[0]:.4f}")
        print(f"AP @ IoU=0.50:      {coco_eval.stats[1]:.4f}")
        print(f"AP @ IoU=0.75:      {coco_eval.stats[2]:.4f}")
        print(f"AP @ small:         {coco_eval.stats[3]:.4f}")
        print(f"AP @ medium:        {coco_eval.stats[4]:.4f}")
        print(f"AP @ large:         {coco_eval.stats[5]:.4f}")
        print(f"AR @ maxDets=1:     {coco_eval.stats[6]:.4f}")
        print(f"AR @ maxDets=10:    {coco_eval.stats[7]:.4f}")
        print(f"AR @ maxDets=100:   {coco_eval.stats[8]:.4f}")
        print(f"AR @ small:         {coco_eval.stats[9]:.4f}")
        print(f"AR @ medium:        {coco_eval.stats[10]:.4f}")
        print(f"AR @ large:         {coco_eval.stats[11]:.4f}")
        print(f"{'='*60}\n")
    
    return {
        'AP': coco_eval.stats[0],
        'AP50': coco_eval.stats[1],
        'AP75': coco_eval.stats[2],
        'APs': coco_eval.stats[3],
        'APm': coco_eval.stats[4],
        'APl': coco_eval.stats[5],
    }


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python evaluate_single.py <pred_json> <gt_json>")
        print("Example: python evaluate_single.py tuiliBIoU_biou_0_0/refined.json data/cityscapes/annotations/instancesonly_filtered_gtFine_val.json")
        sys.exit(1)
    
    pred_json = sys.argv[1]
    gt_json = sys.argv[2]
    
    evaluate_single(pred_json, gt_json)

