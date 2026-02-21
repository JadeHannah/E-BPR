# E-BPR
# Fine-Grained Instance Segmentation Network Based on Adaptive Mask Block Extraction

## Overview

This repository provides the PyTorch implementation of the paper:

**Fine-Grained Instance Segmentation Network Based on Adaptive Mask Block Extraction**

Instance segmentation plays a critical role in high-level vision tasks. However, existing approaches often suffer from coarse mask predictions and inaccurate object boundaries due to limited boundary-sensitive representation.

To address this issue, we propose a fine-grained instance segmentation framework based on an adaptive mask block extraction strategy. The proposed method enhances boundary-aware representation by dynamically extracting and refining mask regions that contain high-frequency contour information. This design improves mask precision while maintaining computational efficiency.

The proposed approach achieves improved boundary refinement performance on standard instance segmentation benchmarks.

## Environment

The code was tested under the following configuration:

- OS: Ubuntu 20.04
- Python: 3.7.16
- PyTorch: 1.7.0
- CUDA: 11.0
- GPU: NVIDIA GPU (≥ 24GB recommended)

## Dataset Preparation

First, you need to generate the instance segmentation results on the Cityscapes training and validation set. Then use the provided script to generate the training set.

sh tools/prepare_dataset.sh \
  maskrcnn_train \
  maskrcnn_val \
  maskrcnn_r50

## Training

DATA_ROOT=maskrcnn_r50/patches \
bash tools/dist_train.sh \
  configs/bpr/hrnet18s_128.py 

## Evaluation

python test.py
The evaluation script reports standard instance segmentation metrics (e.g., AP, AP50, AP75).

## Experimental Results

The proposed method improves boundary accuracy and mask quality compared to baseline models.

Quantitative and qualitative comparisons will be provided upon publication.
