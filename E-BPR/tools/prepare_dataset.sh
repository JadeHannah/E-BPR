#!/bin/bash

coarse_dir_train=$1
coarse_dir_val=$2
prefix=$3

# 可配置参数
IOU_THRESH=${IOU_THRESH:-0.8}
PATCH_SIZE=${PATCH_SIZE:-64}
MAX_INST=${MAX_INST:-20}

IMG_DIR_TRAIN=${IMG_DIR_TRAIN:-'data/cityscapes/leftImg8bit/train'}
IMG_DIR_VAL=${IMG_DIR_VAL:-'data/cityscapes/leftImg8bit/val'}

GT_JSON_TRAIN=${GT_JSON_TRAIN:-'data/cityscapes/annotations/instancesonly_filtered_gtFine_train.json'}
GT_JSON_VAL=${GT_JSON_VAL:-'data/cityscapes/annotations/instancesonly_filtered_gtFine_val.json'}

BPR_ROOT=${BPR_ROOT:-'.'}

coarse_json_train=${prefix}/coarse_train.json
coarse_json_train_filtered=${prefix}/coarse_train_filtered.json
coarse_json_val=${prefix}/coarse_val.json
coarse_json_val_filtered=${prefix}/coarse_val_filtered.json
dataset_dir=${prefix}/patches

set -x
GREEN='\033[0;32m'
END='\033[0m\n'

mkdir -p $prefix

# ========== 1. Training Set ==========
printf ${GREEN}"[1/2] Building training set..."${END}

# Step 1: 预测 mask + txt → COCO-style JSON
python $BPR_ROOT/tools/generate_patch_json_from_pred.py \
    $coarse_dir_train \
    $coarse_json_train

# Step 2: 与 GT IoU 过滤（IoU > 0.5）
python $BPR_ROOT/tools/filter.py \
    $coarse_json_train \
    $GT_JSON_TRAIN \
    $coarse_json_train_filtered

# Step 3: 切图 patch（训练集）
python $BPR_ROOT/tools/split_patches.py \
    $coarse_json_train_filtered \
    $GT_JSON_TRAIN \
    $IMG_DIR_TRAIN \
    $dataset_dir \
    --iou-thresh $IOU_THRESH \
    --mode train \
    --patch-size $PATCH_SIZE \
    --max-inst $MAX_INST


# ========== 2. Validation Set ==========
printf ${GREEN}"[2/2] Building validation set..."${END}

# Step 1: 预测 mask + txt → COCO-style JSON
python $BPR_ROOT/tools/generate_patch_json_from_pred.py \
    $coarse_dir_val \
    $coarse_json_val

# Step 2: 过滤
python $BPR_ROOT/tools/filter.py \
    $coarse_json_val \
    $GT_JSON_VAL \
    $coarse_json_val_filtered

# Step 3: 切图 patch（验证集）
python $BPR_ROOT/tools/split_patches.py \
    $coarse_json_val_filtered \
    $GT_JSON_VAL \
    $IMG_DIR_VAL \
    $dataset_dir \
    --iou-thresh $IOU_THRESH \
    --mode val \
    --patch-size $PATCH_SIZE \
    --max-inst $MAX_INST
