import os
_data_root = os.environ.get('DATA_ROOT')
del os

_base_ = [
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_160k.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoderRefine',
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        type='HRNetRefine',
        norm_cfg=norm_cfg,
        norm_eval=False,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(48, 96)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        type='ASPPDUCHead',
        in_channels=[48, 96, 192, 384],
        in_index=(0, 1, 2, 3),
        channels=256,
        input_transform='multiple_select',  # 多分支输入模式
        low_res_branches=[1, 2, 3],  # 处理1/8, 1/16, 1/32分辨率分支
        dilations=(1, 2, 5),  # ASPP-like膨胀卷积的膨胀率
        target_channels=48,  # 对齐到最高分辨率分支的通道数
        num_classes=2,  # 双通道输出（背景+前景），用于CrossEntropyLoss
        norm_cfg=norm_cfg,
        align_corners=False,
        ignore_index=255,  # 忽略索引（在decode_head级别设置）
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,  # CrossEntropyLoss不需要sigmoid
            loss_weight=1.0)))

# model training and testing settings
train_cfg = dict()
test_cfg = dict(mode='whole')

# Checkpoint and evaluation settings inherit from schedule_160k.py
# checkpoint_config: by_epoch=False, interval=16000 (no save_best)
# evaluation: interval=16000, metric='mIoU'

# Optimizer settings - 使用原论文的学习率（继承自schedule_160k.py: lr=0.01）
# 如果需要显式指定，可以取消下面的注释
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)

# fp16 settings for mixed precision training (saves memory)
# Temporarily disabled due to type mismatch issues
# optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)

# dataset settings
dataset_type = 'RefineDataset'
data_root = _data_root
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='LoadCoarseMask'),
    dict(type='Resize', img_scale=crop_size, ratio_range=(1.0, 1.0)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'coarse_mask']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='LoadCoarseMask'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=crop_size,
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=crop_size, keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'coarse_mask']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/train',
        mask_dir='mask_dir/train',
        ann_dir='ann_dir/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/val',
        mask_dir='mask_dir/val',
        ann_dir='ann_dir/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/val',
        mask_dir='mask_dir/val',
        ann_dir='ann_dir/val',
        pipeline=test_pipeline))