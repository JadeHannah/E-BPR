import json 
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from pycocotools.coco import COCO
import multiprocessing as mp
from fire import Fire
import os  # ✅ 加上这个
from PIL import Image  # ✅ 如果还没加，也加上它
coco = None  # 全局变量用于并行函数中访问


def cal_iou(mask1, mask2):
    """计算两个二值掩码的 IoU"""
    si = np.sum(mask1 & mask2)
    su = np.sum(mask1 | mask2)
    return si / su if su > 0 else 0


def max_iou(inst):
    """对每个预测实例计算其与 GT 的最大 IoU"""
    global coco
    imgid = inst['image_id']
    catid = inst['category_id']

    seg_path = inst.get('segmentation')
    if not seg_path or not os.path.exists(seg_path):
        return 0, inst

    maskdt = np.array(Image.open(seg_path))
    if maskdt.ndim == 3:
        maskdt = maskdt[:, :, 0]
    maskdt = maskdt > 0

    annids = coco.getAnnIds(imgIds=imgid, catIds=catid)
    anns = coco.loadAnns(annids)

    masks = []
    for ann in anns:
        if not ann.get('iscrowd', False):
            mask_gt = coco.annToMask(ann)
            masks.append(mask_gt)

    ious = [cal_iou(maskdt, m) for m in masks]
    miou = max(ious) if ious else 0
    return miou, inst



def filter_iou(bboxs, thresh=0.5):
    """使用多进程过滤掉 IoU 低于阈值的预测实例"""
    out = list()
    with mp.Pool(processes=20) as p:
        with tqdm(total=len(bboxs), desc='Filtering by IoU') as pbar:
            for iou, bbox in p.imap(max_iou, bboxs):
                if iou > thresh:
                    out.append(bbox)
                pbar.update()
    return out


def main(dt_json, gt_json, out_json, thresh=0.5):
    """
    过滤预测 JSON 文件中 IoU < 阈值的实例，保存为完整 COCO 格式 JSON

    Args:
        dt_json (str): COCO 格式预测结果路径
        gt_json (str): GT 标注 JSON 文件路径
        out_json (str): 过滤后结果保存路径
        thresh (float): IoU 阈值（默认 0.5）
    """
    global coco
    coco = COCO(gt_json)

    # 加载预测 JSON
    with open(dt_json, 'r') as f:
        dt_all = json.load(f)

    # 提取 annotations
    if isinstance(dt_all, dict) and 'annotations' in dt_all:
        original_annotations = dt_all['annotations']
    else:
        raise ValueError("Input dt_json must be a COCO-format dict with 'annotations' key.")

    # 过滤
    dt_filtered = filter_iou(original_annotations, thresh)

    # 替换 annotations，保留 images、categories 等原字段
    dt_all['annotations'] = dt_filtered

    # 保存完整 JSON
    with open(out_json, 'w') as f:
        json.dump(dt_all, f)
    print(f"[✔] Saved filtered JSON to {out_json}")


if __name__ == '__main__':
    Fire(main)
