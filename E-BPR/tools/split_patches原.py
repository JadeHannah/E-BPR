import numpy as np
import json
import cv2
import os
import os.path as osp
from tqdm import tqdm
from argparse import ArgumentParser
from pycocotools.coco import COCO
from PIL import Image
import torch
import torch.nn.functional as F
import warnings
import multiprocessing as mp

try:
    from mmdet.ops.nms import nms
    ops = 'mmdet'
except Exception:
    from mmcv.ops.nms import nms
    ops = 'mmcv'

coco = None

def cal_iou(mask1, mask2):
    si = np.sum(mask1 & mask2)
    su = np.sum(mask1 | mask2)
    return si / su if su > 0 else 0

def query_gt_mask(maskdt, coco, imgid, catid):
    annids = coco.getAnnIds(imgIds=imgid, catIds=catid)
    anns = coco.loadAnns(annids)
    masks = [coco.annToMask(ann) for ann in anns if not ann.get('iscrowd', False)]
    ious = [cal_iou(maskdt, m) for m in masks]
    return masks[np.argmax(ious)] if ious else np.zeros(maskdt.shape)

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

def get_dets(maskdt, patch_size, iou_thresh=0.3):
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
    return _force_move_back(dets[inds], *maskdt.shape, patch_size)

def save_patch(pid, img_patch, dt_patch, gt_patch, out_dir):
    fn = f"{pid}.png"
    for subdir, patch in zip(["img_dir", "mask_dir", "ann_dir"],
                             [img_patch, dt_patch, gt_patch]):
        dpath = osp.join(out_dir, subdir, args.mode)
        os.makedirs(dpath, exist_ok=True)
        cv2.imwrite(osp.join(dpath, fn), patch)

def save_details(inst_id, dets, image_name, category, patches, root_dir, mode):
    dets = dets.astype(int)
    xs, ys = dets[:, 0], dets[:, 1]
    ws = dets[:, 2] - dets[:, 0]
    hs = dets[:, 3] - dets[:, 1]
    sdict = dict(
        instance_id=inst_id,
        image_name=image_name,
        category=category,
        patches=patches,
        hlist=list(zip(ys.tolist(), hs.tolist())),
        wlist=list(zip(xs.tolist(), ws.tolist()))
    )
    subroot = osp.join(root_dir, "detail_dir", mode, f"{inst_id}.json")
    os.makedirs(osp.dirname(subroot), exist_ok=True)
    with open(subroot, "w") as f:
        json.dump(sdict, f)

def crop(img, maskdt, maskgt, dets, padding):
    dets = dets.astype(int)[:, :4] + padding
    pd = padding
    img = np.pad(img, ((pd, pd), (pd, pd), (0, 0)))
    maskdt = np.pad(maskdt, pd)
    maskgt = np.pad(maskgt, pd)
    img_patches, dt_patches, gt_patches = [], [], []
    for x1, y1, x2, y2 in dets:
        img_patches.append(img[y1:y2, x1:x2, :])
        dt_patches.append(maskdt[y1:y2, x1:x2])
        gt_patches.append(maskgt[y1:y2, x1:x2])
    return img_patches, dt_patches, gt_patches

def run_inst(i_inst):
    inst_id, inst = i_inst
    global coco
    imgid = inst['image_id']
    catid = inst['category_id']
    imgname = coco.imgs[imgid]["file_name"]

    img_path = osp.join(args.imgs_dir, imgname)
    if not osp.exists(img_path):
        return

    img = cv2.imread(img_path)
    if img is None:
        return

    if img.shape[0] < args.patch_size or img.shape[1] < args.patch_size:
        return

    seg_path = inst.get("segmentation", "")
    if not seg_path or not osp.exists(seg_path):
        return

    mask_raw = np.array(Image.open(seg_path))
    if mask_raw.ndim == 3:
        mask_raw = mask_raw[:, :, 0]
    maskdt = (mask_raw > 0).astype(np.uint8)
    maskgt = query_gt_mask(maskdt, coco, imgid, catid)
    dets = get_dets(maskdt, args.patch_size, args.iou_thresh)
    img_patches, dt_patches, gt_patches = crop(img, maskdt, maskgt, dets, args.padding)

    patchids = []
    for i in range(len(dets)):
        if i >= args.max_inst:
            break
        patchid = inst_id * args.max_inst + i
        save_patch(patchid, img_patches[i], dt_patches[i]*255, gt_patches[i]*255, args.out_dir)
        patchids.append(patchid)

    save_details(inst_id, dets, imgname, coco.cats[catid], patchids, args.out_dir, args.mode)
    print(f"[DEBUG] inst {inst_id} generated {len(dets)} dets, saved {len(patchids)} patches")

def start():
    for d in ["img_dir", "mask_dir", "ann_dir", "detail_dir"]:
        os.makedirs(osp.join(args.out_dir, d, args.mode), exist_ok=True)

    global coco
    coco = COCO(args.gt_json)
    with open(args.dt_json) as f:
        dt = json.load(f)
    if isinstance(dt, dict):
        dt = dt["annotations"]
    if args.sample_inst > 0:
        dt = [dt[i] for i in np.random.choice(len(dt), args.sample_inst, replace=False)]

    with mp.Pool(processes=args.num_proc) as p:
        with tqdm(total=len(dt)) as pbar:
            for _ in p.imap_unordered(run_inst, enumerate(dt)):
                pbar.update()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dt_json', help='Path to coarse JSON')
    parser.add_argument('gt_json', help='Path to GT JSON')
    parser.add_argument('imgs_dir', help='Path to leftImg8bit/{split}/')
    parser.add_argument('out_dir', help='Output dir for patches')
    parser.add_argument('--mode', default='train', choices=['train', 'val'])
    parser.add_argument('--iou-thresh', default=0.25, type=float)
    parser.add_argument('--patch-size', default=64, type=int)
    parser.add_argument('--padding', default=0, type=int)
    parser.add_argument('--num-proc', default=20, type=int)
    parser.add_argument('--max-inst', default=20, type=int)
    parser.add_argument('--sample-inst', default=-1, type=int)
    args = parser.parse_args()

    np.random.seed(2020)
    start()
print(f"[INFO] max_inst = {args.max_inst}")
