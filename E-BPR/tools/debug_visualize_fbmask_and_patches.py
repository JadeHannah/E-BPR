import numpy as np
import cv2
import os
import os.path as osp
import matplotlib.pyplot as plt
from PIL import Image
from argparse import ArgumentParser

import torch
import torch.nn.functional as F

try:
    from mmdet.ops.nms import nms
    ops = 'mmdet'
except Exception:
    from mmcv.ops.nms import nms
    ops = 'mmcv'

def find_float_boundary(maskdt, width=3):
    maskdt = torch.Tensor(maskdt).unsqueeze(0).unsqueeze(0)
    boundary_finder = maskdt.new_ones((1, 1, width, width))
    boundary_mask = F.conv2d(maskdt.permute(1, 0, 2, 3), boundary_finder,
                             stride=1, padding=width//2).permute(1, 0, 2, 3)
    bml = torch.abs(boundary_mask - width*width)
    bms = torch.abs(boundary_mask)
    fbmask = torch.min(bml, bms) / (width*width/2)
    return fbmask[0, 0].numpy()

def _force_move_back(sdets, H, W, patch_size):
    sdets = sdets.copy()
    s = sdets[:, 0] < 0
    sdets[s, 0] = 0
    sdets[s, 2] = patch_size

    s = sdets[:, 1] < 0
    sdets[s, 1] = 0
    sdets[s, 3] = patch_size

    s = sdets[:, 2] >= W
    sdets[s, 0] = W - 1 - patch_size
    sdets[s, 2] = W - 1

    s = sdets[:, 3] >= H
    sdets[s, 1] = H - 1 - patch_size
    sdets[s, 3] = H - 1
    return sdets

def get_dets(maskdt, patch_size, iou_thresh=0.55):
    fbmask = find_float_boundary(maskdt)
    ys, xs = np.where(fbmask)
    scores = fbmask[ys, xs]
    dets = np.stack([xs - patch_size // 2, ys - patch_size // 2,
                     xs + patch_size // 2, ys + patch_size // 2, scores], axis=1)
    if ops == 'mmdet':
        _, inds = nms(dets, iou_thresh)
    else:
        _, inds = nms(np.ascontiguousarray(dets[:, :4], np.float32),
                      np.ascontiguousarray(dets[:, 4], np.float32),
                      iou_thresh)
    return fbmask, _force_move_back(dets[inds], *maskdt.shape, patch_size)

def visualize(mask_path, out_path, patch_size=64, iou_thresh=0.55):
    mask_raw = np.array(Image.open(mask_path))
    if mask_raw.ndim == 3:
        mask_raw = mask_raw[:, :, 0]
    maskdt = (mask_raw > 0).astype(np.uint8)

    fbmask, dets = get_dets(maskdt, patch_size, iou_thresh)

    # 可视化边界热图和补丁框
    H, W = maskdt.shape
    canvas = cv2.cvtColor((maskdt * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for det in dets.astype(int):
        x1, y1, x2, y2 = det[:4]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # 显示并保存
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path + '_mask_with_boxes.png', canvas)
    plt.imsave(out_path + '_fbmask.png', fbmask, cmap='hot')
    print(f"[✔] Saved debug visualization to {out_path}_*.png")
    print(f"Patch proposals generated: {len(dets)}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('mask_path', help='Path to instance mask PNG')
    parser.add_argument('out_path', help='Prefix path to save result')
    parser.add_argument('--patch-size', type=int, default=64)
    parser.add_argument('--iou-thresh', type=float, default=0.55)
    args = parser.parse_args()

    visualize(args.mask_path, args.out_path,
              patch_size=args.patch_size,
              iou_thresh=args.iou_thresh)
