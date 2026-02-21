import os
import os.path as osp
import json
import numpy as np
import pickle as pkl
import multiprocessing as mp
from tqdm import tqdm
from argparse import ArgumentParser
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
from PIL import Image

coco = None
results = None
args = None  # 在 start() 中赋值


def run_inst(i_inst):
    inst_id, inst = i_inst
    global coco, results, args

    if inst_id < 5:
        print(f"[Debug] inst_id {inst_id} type: {type(inst)} content preview: {str(inst)[:300]}")

    # 读取实例 coarse mask（即 segmentation 指向的 PNG 掩码）
    segm_path = inst.get('segmentation', '')
    if not osp.exists(segm_path):
        print(f"[Error] inst_id {inst_id} - segmentation mask not found at path: {segm_path}")
        return inst  # 或 return None，如果后续逻辑能处理

    newmask = np.array(Image.open(segm_path).convert('L')) > 0
    newmask = newmask.astype(np.uint8)

    # load patch proposals
    detail_path = osp.join(args.details_dir, f"{inst_id}.txt")
    if not osp.exists(detail_path):
        print(f"[Warning] inst_id {inst_id} - detail file not found: {detail_path}")
        # 如果没有补丁，将segmentation转换为RLE格式（使用原始coarse mask）
        # 这样json2cityscapes.py才能正确处理
        segm = mask_utils.encode(np.asfortranarray(newmask))
        inst["segmentation"] = {
            "size": list(newmask.shape),
            "counts": segm["counts"].decode("utf8")
        }
        return inst

    with open(detail_path) as f:
        lines = f.readlines()

    patches = []
    hlist = []
    wlist = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7:
            print(f"[Warning] inst_id {inst_id} - invalid line in detail file: {line}")
            continue
        try:
            pid = int(parts[0])
            x, y, w, h = map(int, parts[-4:])
        except ValueError:
            print(f"[Warning] inst_id {inst_id} - value parsing error in line: {line}")
            continue
        patches.append(pid)
        hlist.append((y, h))
        wlist.append((x, w))

    if not patches:
        print(f"[Warning] inst_id {inst_id} - no valid patches found.")
        # 如果没有补丁，使用原始coarse mask并转换为RLE
        segm = mask_utils.encode(np.asfortranarray(newmask))
        inst["segmentation"] = {
            "size": list(newmask.shape),
            "counts": segm["counts"].decode("utf8")
        }
        return inst

    # reassemble mask
    newmask_refined = np.zeros_like(newmask, dtype=np.float32)
    newmask_count = np.zeros_like(newmask, dtype=np.float32)

    for j, pid in enumerate(patches):
        y, h = hlist[j]
        x, w = wlist[j]
        patch_mask = results.get(pid, None)
        if patch_mask is None:
            print(f"[Warning] inst_id {inst_id} - missing patch mask for pid {pid}")
            continue
        if args.padding:
            p = args.padding
            patch_mask = patch_mask[p:-p, p:-p]
        newmask_refined[y:y + h, x:x + w] += patch_mask
        newmask_count[y:y + h, x:x + w] += 1

    s = newmask_count > 0
    if s.sum() == 0:
        # 如果没有任何区域被refined，使用原始coarse mask
        print(f"[Warning] inst_id {inst_id} - no refined regions, using coarse mask")
        newmask = np.array(Image.open(segm_path).convert('L')) > 0
        newmask = newmask.astype(np.uint8)
    else:
        newmask_refined[s] /= newmask_count[s]
        newmask[s] = (newmask_refined[s] > 0.5).astype(np.uint8)

    # update segmentation with RLE encoding
    segm = mask_utils.encode(np.asfortranarray(newmask))
    inst["segmentation"] = {
        "size": list(newmask.shape),
        "counts": segm["counts"].decode("utf8")
    }

    return inst


def start():
    global coco, results, args

    coco = COCO(args.gt_json)

    with open(args.dt_json, 'r') as f:
        dt = json.load(f)

    print(f"[Debug] dt keys: {list(dt.keys())}")
    annotations = dt.get('annotations', [])
    print(f"[Debug] dt['annotations'] type: {type(annotations)}, length: {len(annotations)}")
    if annotations:
        print(f"[Debug] dt['annotations'][0] preview: {str(annotations[0])[:300]}")

    # load network's output patch masks
    with open(args.res_pkl, 'rb') as f:
        _res = pkl.load(f)  # img_infos, masks
    img_infos, masks = _res
    
    # 构建results字典，提取patch ID
    # seg_map的值是文件名（如'1002000001.png'），文件名本身就是patch ID（去掉.png后缀）
    results = {}
    failed_count = 0
    for img_info, mask in zip(img_infos, masks):
        try:
            # 方法1：从filename提取（优先）
            if 'filename' in img_info:
                filename = img_info['filename']
                pid = int(filename.replace('.png', ''))
                results[pid] = mask
            # 方法2：从seg_map提取（备用）
            elif 'ann' in img_info and 'seg_map' in img_info['ann']:
                seg_map = img_info['ann']['seg_map']
                pid = int(seg_map.split('.')[0])
                results[pid] = mask
            else:
                raise KeyError("Neither 'filename' nor 'ann.seg_map' found")
        except (KeyError, ValueError, AttributeError) as e:
            failed_count += 1
            if failed_count <= 5:  # 只打印前5个错误
                print(f"[Warning] Failed to extract PID from img_info: {e}")
                print(f"  img_info keys: {list(img_info.keys())}")
                if 'ann' in img_info:
                    print(f"  ann keys: {list(img_info['ann'].keys())}")
                    if 'seg_map' in img_info['ann']:
                        print(f"  seg_map value: {img_info['ann']['seg_map']}")
                if 'filename' in img_info:
                    print(f"  filename value: {img_info['filename']}")
    
    if failed_count > 0:
        print(f"[Warning] Failed to extract PID from {failed_count} items in PKL")
    
    print(f"[Info] Loaded {len(results)} patch masks from PKL")
    if results:
        print(f"[Info] PKL PID range: {min(results.keys())} - {max(results.keys())}")

    refined_res = []
    with mp.Pool(processes=args.num_proc) as pool:
        with tqdm(total=len(annotations)) as pbar:
            for r in pool.imap_unordered(run_inst, enumerate(annotations)):
                if r is not None:
                    # 确保segmentation是RLE格式，而不是文件路径字符串
                    if isinstance(r.get('segmentation'), str):
                        # 如果仍然是字符串，说明处理失败，使用原始mask转换为RLE
                        segm_path = r.get('segmentation', '')
                        if osp.exists(segm_path):
                            mask = np.array(Image.open(segm_path).convert('L')) > 0
                            mask = mask.astype(np.uint8)
                            segm = mask_utils.encode(np.asfortranarray(mask))
                            r["segmentation"] = {
                                "size": list(mask.shape),
                                "counts": segm["counts"].decode("utf8")
                            }
                        else:
                            print(f"[Error] inst_id {r.get('id', 'unknown')} - segmentation path not found: {segm_path}")
                            continue  # 跳过这个annotation
                    refined_res.append(r)
                pbar.update()

    # 更新原 dt json 的 annotations 部分
    dt['annotations'] = refined_res
    with open(args.out_json, 'w') as f:
        json.dump(dt, f)


if __name__ == "__main__":
    parser = ArgumentParser(description='Reassemble the refined patches into json file.')
    parser.add_argument('dt_json', help='path to coarse masks (json format)')
    parser.add_argument('gt_json', help='path to ground-truth annotations (json format)')
    parser.add_argument('res_pkl', help='path to network output (pkl format)')
    parser.add_argument('details_dir', help='path to detail_dir/')
    parser.add_argument('out_json', help='where to save the output refined masks')
    parser.add_argument('--padding', type=int, default=0, help='padding size to remove from each patch')
    parser.add_argument('--num-proc', type=int, default=20, help='number of processes')
    args = parser.parse_args()

    start()
