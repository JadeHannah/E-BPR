#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BIoU阈值消融实验脚本
用于批量运行不同BIoU阈值的推理实验
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path


def check_result_complete(prefix):
    """
    检查推理结果是否已完成
    
    Args:
        prefix: 输出目录路径
    
    Returns:
        bool: 如果最终结果存在则返回True
    """
    prefix_path = Path(prefix)
    # 检查最终输出文件是否存在
    final_output = prefix_path / 'refined.json'
    return final_output.exists()


def run_inference(biou_thresh, config, ckpt, coarse_dir, prefix_base, 
                  img_dir, gt_json, bpr_root, gpus=1, biou_dilation_ratio=0.005, 
                  skip_completed=True):
    """
    运行单个BIoU阈值的推理
    
    Args:
        biou_thresh: BIoU阈值
        config: 配置文件路径
        ckpt: 权重文件路径
        coarse_dir: 粗分割结果目录
        prefix_base: 输出目录基础路径
        img_dir: 图像目录
        gt_json: GT JSON文件路径
        bpr_root: BPR根目录
        gpus: GPU数量
        biou_dilation_ratio: BIoU边界膨胀比例
        skip_completed: 如果结果已完成则跳过
    """
    # 为每个阈值创建独立的输出目录
    prefix = f"{prefix_base}_biou_{biou_thresh:.3f}".replace('.', '_')
    
    print(f"\n{'='*80}")
    print(f"Running inference with BIoU threshold: {biou_thresh}")
    print(f"Output directory: {prefix}")
    print(f"{'='*80}\n")
    
    # 检查是否已完成
    if skip_completed and check_result_complete(prefix):
        print(f"✓ Result already exists for BIoU threshold: {biou_thresh}")
        print(f"  Skipping... (use --no-skip to force re-run)\n")
        return True
    
    # 设置环境变量
    env = os.environ.copy()
    env['IOU_THRESH'] = str(biou_thresh)
    env['NMS_THRESH'] = '0.55'  # 固定NMS阈值为0.55
    env['USE_BIOU'] = '1'
    env['BIOU_DILATION_RATIO'] = str(biou_dilation_ratio)
    env['IMG_DIR'] = img_dir
    env['GT_JSON'] = gt_json
    env['BPR_ROOT'] = bpr_root
    env['GPUS'] = str(gpus)
    
    # 构建命令
    cmd = [
        'bash',
        os.path.join(bpr_root, 'tools', 'inference.sh'),
        config,
        ckpt,
        coarse_dir,
        prefix
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Environment variables:")
    print(f"  IOU_THRESH={env['IOU_THRESH']}")
    print(f"  USE_BIOU={env['USE_BIOU']}")
    print(f"  BIOU_DILATION_RATIO={env['BIOU_DILATION_RATIO']}")
    print(f"  GPUS={env['GPUS']}\n")
    
    # 运行推理
    try:
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            cwd=bpr_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        print(f"\n✓ Successfully completed inference with BIoU threshold: {biou_thresh}")
        print(f"  Output saved to: {prefix}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed to run inference with BIoU threshold: {biou_thresh}")
        print(f"  Error output:\n{e.stdout}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Run BIoU threshold sweep ablation experiments'
    )
    parser.add_argument('config', help='Path to config file')
    parser.add_argument('ckpt', help='Path to checkpoint file')
    parser.add_argument('coarse_dir', help='Path to coarse segmentation results directory')
    parser.add_argument('prefix_base', help='Base path for output directories')
    parser.add_argument(
        '--biou-threshs',
        nargs='+',
        type=float,
        default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
        help='List of BIoU thresholds to test (default: 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5, because BIoU is usually stricter than IoU)'
    )
    parser.add_argument(
        '--biou-dilation-ratio',
        type=float,
        default=0.005,
        help='BIoU boundary dilation ratio (default: 0.005)'
    )
    parser.add_argument(
        '--img-dir',
        default='data/cityscapes/leftImg8bit/val',
        help='Path to image directory (default: data/cityscapes/leftImg8bit/val)'
    )
    parser.add_argument(
        '--gt-json',
        default='data/cityscapes/annotations/instancesonly_filtered_gtFine_val.json',
        help='Path to GT JSON file'
    )
    parser.add_argument(
        '--bpr-root',
        default='.',
        help='BPR root directory (default: .)'
    )
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='Number of GPUs (default: 1)'
    )
    parser.add_argument(
        '--start-from',
        type=float,
        default=None,
        help='Start from a specific threshold (useful for resuming)'
    )
    parser.add_argument(
        '--no-skip',
        action='store_true',
        help='Do not skip completed results (force re-run)'
    )
    
    args = parser.parse_args()
    
    # 验证路径
    bpr_root = Path(args.bpr_root).resolve()
    if not (bpr_root / 'tools' / 'inference.sh').exists():
        print(f"Error: inference.sh not found in {bpr_root / 'tools'}")
        sys.exit(1)
    
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = bpr_root / config_path
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = bpr_root / ckpt_path
    if not ckpt_path.exists():
        print(f"Error: Checkpoint file not found: {ckpt_path}")
        sys.exit(1)
    
    # 处理阈值列表
    biou_threshs = sorted(args.biou_threshs)
    if args.start_from is not None:
        biou_threshs = [t for t in biou_threshs if t >= args.start_from]
        print(f"Starting from threshold: {args.start_from}")
    
    print(f"\n{'='*80}")
    print(f"BIoU Threshold Sweep Experiment")
    print(f"{'='*80}")
    print(f"Config: {config_path}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Coarse dir: {args.coarse_dir}")
    print(f"Output base: {args.prefix_base}")
    print(f"BIoU thresholds: {biou_threshs}")
    print(f"BIoU dilation ratio: {args.biou_dilation_ratio}")
    print(f"GPUs: {args.gpus}")
    print(f"{'='*80}\n")
    
    # 运行每个阈值的推理
    results = []
    for biou_thresh in biou_threshs:
        success = run_inference(
            biou_thresh=biou_thresh,
            config=str(config_path),
            ckpt=str(ckpt_path),
            coarse_dir=args.coarse_dir,
            prefix_base=args.prefix_base,
            img_dir=args.img_dir,
            gt_json=args.gt_json,
            bpr_root=str(bpr_root),
            gpus=args.gpus,
            biou_dilation_ratio=args.biou_dilation_ratio,
            skip_completed=not args.no_skip
        )
        results.append((biou_thresh, success))
    
    # 打印总结
    print(f"\n{'='*80}")
    print(f"Experiment Summary")
    print(f"{'='*80}")
    for biou_thresh, success in results:
        status = "✓ Success" if success else "✗ Failed"
        print(f"BIoU={biou_thresh:.3f}: {status}")
    print(f"{'='*80}\n")
    
    # 统计
    success_count = sum(1 for _, s in results if s)
    total_count = len(results)
    print(f"Total: {success_count}/{total_count} successful")
    
    if success_count < total_count:
        sys.exit(1)


if __name__ == '__main__':
    main()

