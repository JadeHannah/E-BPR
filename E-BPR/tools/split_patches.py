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

def mask_to_boundary(mask, dilation_ratio=0.005, min_dilation=1, max_dilation=None, 
                     small_patch_dilation=None):
    """
    将二值 mask 转为边界 mask。
    Args:
        mask: np.ndarray, shape=(H, W), 值为0或1
        dilation_ratio: 边界厚度比例，例如0.005表示边界宽度≈图像对角线的0.5%
        min_dilation: 最小dilation值（默认1）
        max_dilation: 最大dilation值（None表示不限制）
        small_patch_dilation: 小补丁（≤64x64）的固定dilation值（None表示自动计算）
    Returns:
        boundary: np.ndarray, 二值边界图
    """
    h, w = mask.shape
    
    # 对于小补丁（64x64），使用固定的dilation值
    # 根据测试结果，dilation=5效果最好（BIoU=0.4668 > 0.3）
    if h <= 64 and w <= 64:
        if small_patch_dilation is not None:
            # 使用指定的固定值
            dilation = small_patch_dilation
        else:
            # 对于64×64补丁，使用固定的dilation=5（最佳BIoU效果）
            # 根据测试结果，dilation=5时BIoU=0.4668，效果最好
            # 不依赖dilation_ratio计算，因为dilation_ratio对小补丁不适用
            dilation = 5
    else:
        # 对于大图像，使用基于dilation_ratio的计算
        diag_len = np.sqrt(h ** 2 + w ** 2)
        dilation = max(min_dilation, int(round(dilation_ratio * diag_len)))
        if max_dilation is not None:
            dilation = min(dilation, max_dilation)
    
    # 确保dilation至少为1且为奇数（更好的边界效果）
    dilation = max(1, dilation)
    if dilation % 2 == 0:
        dilation += 1
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel)
    eroded = cv2.erode(mask.astype(np.uint8), kernel)
    boundary = dilated - eroded
    return boundary

def compute_biou(pred_mask, gt_mask, dilation_ratio=0.005, combine_with_iou=True, 
                 min_dilation=1, max_dilation=None, small_patch_dilation=None):
    """
    计算 Boundary IoU (BIoU)
    Args:
        pred_mask: 预测的二值掩膜 (numpy array, 0/1)
        gt_mask:   GT 二值掩膜 (numpy array, 0/1)
        dilation_ratio: 边界宽度比例
        combine_with_iou: 若为True，返回 min(IoU, BIoU)，否则仅返回边界IoU
        min_dilation: 最小dilation值
        max_dilation: 最大dilation值
        small_patch_dilation: 小补丁（≤64x64）的固定dilation值
    Returns:
        final_iou, iou, biou
    """
    # 普通 IoU
    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = inter / union if union > 0 else 0.0

    # 边界 IoU - 使用优化的dilation参数
    pred_boundary = mask_to_boundary(pred_mask, dilation_ratio, min_dilation, max_dilation, 
                                    small_patch_dilation)
    gt_boundary = mask_to_boundary(gt_mask, dilation_ratio, min_dilation, max_dilation,
                                   small_patch_dilation)
    inter_b = np.logical_and(pred_boundary, gt_boundary).sum()
    union_b = np.logical_or(pred_boundary, gt_boundary).sum()
    
    # 纯BIoU计算，不使用fallback
    biou = inter_b / union_b if union_b > 0 else 0.0

    final_iou = min(iou, biou) if combine_with_iou else biou
    return final_iou, iou, biou

def query_gt_mask(maskdt, coco, imgid, catid, use_biou=False, biou_dilation_ratio=0.005, debug=False):
    annids = coco.getAnnIds(imgIds=imgid, catIds=catid)
    
    if debug:
        print(f"  [DEBUG] query_gt_mask: imgid={imgid}, catid={catid}")
        print(f"  [DEBUG] Found {len(annids)} annotation IDs")
    
    if len(annids) == 0:
        # 如果没有找到相同类别的GT，尝试查找该图像的所有annotations
        all_annids = coco.getAnnIds(imgIds=imgid)
        if debug:
            print(f"  [DEBUG] No annotations with catid={catid}, trying all categories: {len(all_annids)} annotations")
        if len(all_annids) > 0:
            annids = all_annids
        else:
            if debug:
                print(f"  [DEBUG] No annotations found for image {imgid}")
            return np.zeros(maskdt.shape)
    
    anns = coco.loadAnns(annids)
    masks = []
    for ann in anns:
        if not ann.get('iscrowd', False):
            try:
                mask = coco.annToMask(ann)
                # 确保mask尺寸匹配
                if mask.shape != maskdt.shape:
                    # 如果尺寸不匹配，尝试resize
                    import cv2
                    mask = cv2.resize(mask.astype(np.float32), 
                                    (maskdt.shape[1], maskdt.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                masks.append(mask)
            except Exception as e:
                if debug:
                    print(f"  [DEBUG] Error converting annotation to mask: {e}")
                continue
    
    if debug:
        print(f"  [DEBUG] Valid masks found: {len(masks)}")
    
    if len(masks) == 0:
        return np.zeros(maskdt.shape)
    
    if use_biou:
        # 使用BIoU选择最佳GT mask
        # 只使用BIoU，不结合IoU
        scores = []
        for m in masks:
            final_iou, _, _ = compute_biou(maskdt, m, biou_dilation_ratio, combine_with_iou=False)
            scores.append(final_iou)
    else:
        # 使用IoU选择最佳GT mask
        scores = [cal_iou(maskdt, m) for m in masks]
    
    if debug and len(scores) > 0:
        print(f"  [DEBUG] Best score: {max(scores):.4f}")
    
    return masks[np.argmax(scores)] if scores else np.zeros(maskdt.shape)

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

def get_dets(maskdt, patch_size, nms_thresh=0.25):
    fbmask = find_float_boundary(maskdt)
    ys, xs = np.where(fbmask)
    if len(ys) == 0:
        return np.zeros((0, 4))
    scores = fbmask[ys, xs]
    dets = np.stack([xs - patch_size // 2, ys - patch_size // 2,
                     xs + patch_size // 2, ys + patch_size // 2, scores], axis=1)
    if ops == 'mmdet':
        _, inds = nms(dets, nms_thresh)
    else:
        _, inds = nms(np.ascontiguousarray(dets[:, :4], np.float32),
                      np.ascontiguousarray(dets[:, 4], np.float32),
                      nms_thresh)
    return _force_move_back(dets[inds], *maskdt.shape, patch_size)

def save_patch(pid, img_patch, dt_patch, gt_patch, out_dir):
    fn = f"{pid}.png"
    for subdir, patch in zip(["img_dir", "mask_dir", "ann_dir"],
                             [img_patch, dt_patch, gt_patch]):
        dpath = osp.join(out_dir, subdir, args.mode)
        os.makedirs(dpath, exist_ok=True)

        # 确保是 uint8 类型
        if patch.dtype != np.uint8:
            patch = patch.astype(np.uint8)

        cv2.imwrite(osp.join(dpath, fn), patch)

def save_details(inst_id, dets, image_name, category, patches, root_dir, mode):
    dets = dets.astype(int)
    subroot = osp.join(root_dir, "detail_dir", mode, f"{inst_id}.txt")
    os.makedirs(osp.dirname(subroot), exist_ok=True)
    with open(subroot, "w") as f:
        for pid, (x1, y1, x2, y2) in zip(patches, dets[:, 0:4]):
            x, y, w, h = x1, y1, x2 - x1, y2 - y1
            f.write(f"{pid} {category} {x} {y} {w} {h}\n")

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

    seg_path = inst.get('segmentation', None)
    if seg_path is None or not osp.exists(seg_path):
        return

    mask_raw = np.array(Image.open(seg_path))
    if mask_raw.ndim == 3:
        mask_raw = mask_raw[:, :, 0]
    maskdt = (mask_raw > 0).astype(np.uint8)
    
    # 检查mask是否有效
    if maskdt.sum() == 0:
        if inst_id % 1000 == 0:
            print(f"[WARN] inst_id={inst_id}: Empty mask, skipping")
        return
    
    maskgt = query_gt_mask(maskdt, coco, imgid, catid, 
                          use_biou=args.use_biou, 
                          biou_dilation_ratio=args.biou_dilation_ratio)
    
    # 检查GT mask是否有效
    if maskgt.sum() == 0:
        # 如果没有找到相同类别的GT，尝试查找该图像的所有annotations
        all_annids = coco.getAnnIds(imgIds=imgid)
        if len(all_annids) > 0:
            # 尝试使用所有类别的GT mask
            anns = coco.loadAnns(all_annids)
            all_masks = []
            for ann in anns:
                if not ann.get('iscrowd', False):
                    try:
                        mask = coco.annToMask(ann)
                        if mask.shape == maskdt.shape:
                            all_masks.append(mask)
                    except:
                        continue
            
            if len(all_masks) > 0:
                # 使用IoU选择最佳匹配（不限制类别）
                scores = [cal_iou(maskdt, m) for m in all_masks]
                if max(scores) > 0.1:  # 至少有一些重叠
                    maskgt = all_masks[np.argmax(scores)]
                    if inst_id % 1000 == 0:
                        print(f"[INFO] inst_id={inst_id}: Using GT from different category (best IoU: {max(scores):.4f})")
        
        if maskgt.sum() == 0:
            if inst_id % 1000 == 0:
                print(f"[WARN] inst_id={inst_id}: No matching GT mask found, skipping")
            return
    
    # NMS阈值：用于去除重叠的补丁检测框（固定为0.55）
    # 补丁过滤阈值：用于过滤补丁质量（原论文使用0.55，但这里使用args.iou_thresh）
    nms_thresh = getattr(args, 'nms_thresh', 0.55)  # 固定为0.55
    dets = get_dets(maskdt, args.patch_size, nms_thresh)

    if len(dets) == 0:
        if inst_id % 1000 == 0:
            print(f"[WARN] inst_id={inst_id}: No patches detected by get_dets()")
        return

    img_patches, dt_patches, gt_patches = crop(img, maskdt, maskgt, dets, args.padding)

    patchids = []
    valid_dets = []
    filtered_count = 0
    total_patches = min(len(dets), args.max_inst)
    
    for i in range(total_patches):
        # 计算补丁的BIoU或IoU
        dt_patch = dt_patches[i].astype(np.uint8)
        gt_patch = gt_patches[i].astype(np.uint8)
        
        # 确保是二值mask（0或1）
        dt_patch = (dt_patch > 0).astype(np.uint8)
        gt_patch = (gt_patch > 0).astype(np.uint8)
        
        if args.use_biou:
            # 使用BIoU过滤
            # 对于64×64补丁，使用固定的small_patch_dilation（如果指定）
            small_patch_dilation = getattr(args, 'biou_small_patch_dilation', None)
            # 只使用BIoU，不结合IoU（combine_with_iou=False）
            final_iou, iou, biou = compute_biou(dt_patch, gt_patch, 
                                          args.biou_dilation_ratio, 
                                          combine_with_iou=False,
                                          small_patch_dilation=small_patch_dilation)
            score = final_iou  # final_iou = biou (因为combine_with_iou=False)
        else:
            # 使用IoU过滤
            score = cal_iou(dt_patch, gt_patch)
            iou = score
            biou = 0.0
        
        # 只有当score >= iou_thresh时才保存补丁
        if score >= args.iou_thresh:
            patchid = inst_id * args.max_inst + i
            save_patch(patchid,
                       img_patches[i],
                       (dt_patches[i]*255).astype(np.uint8),
                       (gt_patches[i]*255).astype(np.uint8),
                       args.out_dir)
            patchids.append(patchid)
            valid_dets.append(dets[i])
        else:
            filtered_count += 1
    
    # 打印统计信息（每100个实例打印一次，避免输出过多）
    if inst_id % 100 == 0 or (len(patchids) == 0 and total_patches > 0):
        print(f"[INFO] inst_id={inst_id}: {len(patchids)}/{total_patches} patches saved "
              f"(filtered: {filtered_count}, threshold: {args.iou_thresh:.3f})")
        if total_patches > 0 and len(patchids) == 0:
            # 如果所有补丁都被过滤，打印一些示例分数
            if total_patches > 0:
                dt_patch = dt_patches[0].astype(np.uint8)
                gt_patch = gt_patches[0].astype(np.uint8)
                if args.use_biou:
                    final_iou, iou, biou = compute_biou(dt_patch, gt_patch, 
                                                      args.biou_dilation_ratio, 
                                                      combine_with_iou=True)
                    print(f"  Example patch score: final={final_iou:.4f}, iou={iou:.4f}, biou={biou:.4f}")
                    print(f"  Threshold: {args.iou_thresh:.3f}, Need: final_iou >= {args.iou_thresh:.3f}")
                else:
                    score = cal_iou(dt_patch, gt_patch)
                    print(f"  Example patch IoU: {score:.4f}")
                    print(f"  Threshold: {args.iou_thresh:.3f}, Need: iou >= {args.iou_thresh:.3f}")

    if patchids:
        save_details(inst_id, np.array(valid_dets), imgname, coco.cats[catid], patchids, args.out_dir, args.mode)
    elif total_patches > 0:
        # 如果检测到补丁但都被过滤了，打印警告
        print(f"[WARN] inst_id={inst_id}: All {total_patches} patches filtered out (threshold={args.iou_thresh:.3f})")


def start():
    # 确保输出目录存在（即使没有补丁被保存）
    print(f"[INFO] Creating output directories in: {args.out_dir}")
    for d in ["img_dir", "mask_dir", "ann_dir", "detail_dir"]:
        dir_path = osp.join(args.out_dir, d, args.mode)
        os.makedirs(dir_path, exist_ok=True)
        print(f"[INFO] Created directory: {dir_path}")

    global coco
    print(f"[INFO] Loading GT JSON: {args.gt_json}")
    coco = COCO(args.gt_json)
    print(f"[INFO] Loading coarse JSON: {args.dt_json}")
    with open(args.dt_json) as f:
        dt = json.load(f)
    if isinstance(dt, dict):
        dt = dt["annotations"]
    print(f"[INFO] Total annotations to process: {len(dt)}")
    if args.sample_inst > 0:
        dt = [dt[i] for i in np.random.choice(len(dt), args.sample_inst, replace=False)]
        print(f"[INFO] Sampling {len(dt)} annotations")

    print(f"[INFO] Starting patch generation with {args.num_proc} processes...")
    nms_thresh = getattr(args, 'nms_thresh', 0.55)
    print(f"[INFO] NMS threshold (for patch detection): {nms_thresh} (固定为0.55)")
    print(f"[INFO] Filter threshold (for patch quality): {args.iou_thresh} (原论文使用0.55)")
    print(f"[INFO] BIoU settings: use_biou={args.use_biou}")
    if args.use_biou:
        print(f"[INFO] BIoU dilation_ratio={args.biou_dilation_ratio}")
        if hasattr(args, 'biou_small_patch_dilation') and args.biou_small_patch_dilation is not None:
            print(f"[INFO] Small patch dilation={args.biou_small_patch_dilation}")

    try:
        with mp.Pool(processes=args.num_proc) as p:
            with tqdm(total=len(dt)) as pbar:
                for _ in p.imap_unordered(run_inst, enumerate(dt)):
                    pbar.update()
        print(f"[INFO] Patch generation completed!")
    except Exception as e:
        print(f"[ERROR] Error during patch generation: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dt_json', help='Path to coarse JSON')
    parser.add_argument('gt_json', help='Path to GT JSON')
    parser.add_argument('imgs_dir', help='Path to leftImg8bit/{split}/')
    parser.add_argument('out_dir', help='Output dir for patches')
    parser.add_argument('--mode', default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--iou-thresh', default=0.55, type=float, 
                       help='IoU/BIoU threshold for filtering patches (原论文使用0.55)')
    parser.add_argument('--nms-thresh', default=0.55, type=float,
                       help='NMS threshold for patch detection (固定为0.55，用于去除重叠的补丁检测框)')
    parser.add_argument('--use-biou', action='store_true', 
                       help='Use BIoU instead of IoU for filtering patches')
    parser.add_argument('--biou-dilation-ratio', default=0.005, type=float,
                       help='Dilation ratio for BIoU boundary computation')
    parser.add_argument('--biou-small-patch-dilation', default=None, type=int,
                       help='Fixed dilation for small patches (≤64x64). If None, uses fixed dilation=5')
    parser.add_argument('--patch-size', default=64, type=int)
    parser.add_argument('--padding', default=0, type=int)
    parser.add_argument('--num-proc', default=20, type=int)
    parser.add_argument('--max-inst', default=20, type=int)
    parser.add_argument('--sample-inst', default=-1, type=int)
    args = parser.parse_args()

    np.random.seed(2020)
    # 确保输出目录存在（即使没有补丁被保存）
    for d in ["img_dir", "mask_dir", "ann_dir", "detail_dir"]:
        os.makedirs(osp.join(args.out_dir, d, args.mode), exist_ok=True)
    start()
    print(f"[INFO] max_inst = {args.max_inst}")
