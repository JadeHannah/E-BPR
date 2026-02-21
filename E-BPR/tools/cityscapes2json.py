import json
import os
import numpy as np
from tqdm import tqdm
from fire import Fire

class Cityscape:
    def __init__(self, gt_file, dt_root):
        self.gt_file = gt_file
        self.dt_root = dt_root
        self._img2id, self._id2img = self._build_gt()
        self._id2annos = self._build()

    def _build_gt(self):
        img2id, id2img = dict(), dict()
        with open(self.gt_file) as f:
            imgs = json.load(f)['images']
            for im in imgs:
                base_name = os.path.basename(im['file_name'])  # e.g. aachen_000003_000019_leftImg8bit.png
                img_name = base_name.split('_leftImg8bit')[0]  # e.g. aachen_000003_000019
                img2id[img_name] = im['id']
                id2img[im['id']] = im['file_name']
        print(f'[INFO] Loaded {len(img2id)} images from GT.')
        return img2id, id2img

    def _build(self):
        id2annos = dict()
        total_files = 0
        total_annos = 0

        for root, _, files in os.walk(self.dt_root):
            for af in files:
                if not af.endswith('_pred.txt'):
                    continue
                total_files += 1
                img_name = af.replace('_pred.txt', '')
                img_name = img_name.split('_index.txt')[0]
                if img_name not in self._img2id:
                    print(f'[WARN] Prediction file "{af}" does not match any GT image id. Skipping.')
                    continue
                img_id = self._img2id[img_name]

                pred_path = os.path.join(root, af)
                annos = []
                with open(pred_path, 'r') as f:
                    for idx, line in enumerate(f):
                        parts = line.strip().split()
                        if len(parts) < 3:
                            continue
                        
                        # 支持两种格式：
                        # 格式1: mask_filename category_id score (3个字段)
                        # 格式2: bbox[4] category_id score (6个字段)
                        
                        if len(parts) == 3:
                            # 格式1: mask_filename category_id score
                            mask_filename = parts[0]
                            category_id = int(parts[1])
                            score = float(parts[2])
                            mask_path = os.path.join(root, mask_filename)
                            
                            if not os.path.exists(mask_path):
                                print(f'[WARN] Missing mask file: {mask_path}, skipping annotation')
                                continue
                            
                            # 从mask文件读取bbox
                            try:
                                import cv2
                                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                                if mask is None:
                                    print(f'[WARN] Cannot read mask file: {mask_path}, skipping')
                                    continue
                                # 计算bbox
                                rows = np.any(mask > 0, axis=1)
                                cols = np.any(mask > 0, axis=0)
                                if not np.any(rows) or not np.any(cols):
                                    print(f'[WARN] Empty mask in: {mask_path}, skipping')
                                    continue
                                y_min, y_max = np.where(rows)[0][[0, -1]]
                                x_min, x_max = np.where(cols)[0][[0, -1]]
                                bbox = [float(x_min), float(y_min), float(x_max), float(y_max)]
                            except Exception as e:
                                print(f'[WARN] Error processing mask {mask_path}: {e}, skipping')
                                continue
                        elif len(parts) >= 6:
                            # 格式2: bbox[4] category_id score
                            bbox = [float(x) for x in parts[:4]]
                            category_id = int(parts[4])
                            score = float(parts[5])

                            # 推测 mask 命名方式：aachen_000003_000019_1.png 等
                            mask_filename = af.replace('_pred.txt', f'_{idx + 1}.png')
                            mask_path = os.path.join(root, mask_filename)
                            if not os.path.exists(mask_path):
                                print(f'[WARN] Missing mask file: {mask_path}, skipping annotation')
                                continue
                        else:
                            continue

                        annos.append({
                            'bbox': bbox,
                            'category_id': category_id,
                            'score': score,
                            'segmentation': mask_path
                        })

                id2annos[img_id] = annos
                total_annos += len(annos)
                print(f'[INFO] Parsed {len(annos)} annotations from {pred_path}')

        print(f'[INFO] Total prediction files processed: {total_files}')
        print(f'[INFO] Total annotations collected: {total_annos}')
        return id2annos


def main(dt_root, gt_json, out_json):
    cs = Cityscape(gt_json, dt_root)
    out = {
        'images': [],
        'annotations': [],
        'categories': []
    }
    with open(gt_json) as f:
        gt = json.load(f)
        out['images'] = gt['images']
        if 'categories' in gt:
            out['categories'] = gt['categories']

    ann_id = 1
    for img_id, annos in cs._id2annos.items():
        for ann in annos:
            bbox = ann['bbox']
            category_id = ann['category_id']
            score = ann['score']
            segmentation = ann.get('segmentation', None)

            annotation = {
                'id': ann_id,
                'image_id': img_id,
                'category_id': category_id,
                'bbox': bbox,
                'score': score,
                'iscrowd': 0,
                'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            }

            if segmentation is not None:
                annotation['segmentation'] = segmentation

            out['annotations'].append(annotation)
            ann_id += 1

    with open(out_json, 'w') as f:
        json.dump(out, f)
    print(f'[INFO] Saved converted json to {out_json}')


if __name__ == '__main__':
    Fire(main)
