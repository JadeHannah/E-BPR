import os
import json
import glob
from collections import defaultdict
import argparse

def generate_json(pred_root, out_json_path):
    image_id_map = {}
    annotations = []
    images = []
    ann_id = 1
    img_id = 1

    city_dirs = sorted(glob.glob(os.path.join(pred_root, '*')))

    for city_dir in city_dirs:
        if not os.path.isdir(city_dir):
            continue
        txt_files = sorted(glob.glob(os.path.join(city_dir, '*_pred.txt')))
        for txt_file in txt_files:
            base = os.path.basename(txt_file).replace('_pred.txt', '')
            with open(txt_file, 'r') as f:
                lines = f.readlines()
                image_file = f'{base}.png'
                image_id_map[base] = img_id
                images.append({
                    'file_name': f'{city_dir}/{image_file}',
                    'id': img_id,
                    'width': 2048,
                    'height': 1024
                })
                for line in lines:
                    mask_file, category_id, score = line.strip().split()
                    annotations.append({
                        'id': ann_id,
                        'image_id': img_id,
                        'category_id': int(category_id),
                        'segmentation': os.path.abspath(os.path.join(city_dir, mask_file)),
                        'score': float(score),
                        'iscrowd': 0,
                        'bbox': [0, 0, 0, 0],  # optional
                        'area': 0              # optional
                    })
                    ann_id += 1
                img_id += 1

    categories = [
        {'id': i, 'name': f'class_{i}'} for i in range(1, 35)
    ]

    result = {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }

    with open(out_json_path, 'w') as f:
        json.dump(result, f)
    print(f"[✔] Saved converted prediction JSON to {out_json_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pred_dir', help='Directory with instance masks and *_pred.txt files')
    parser.add_argument('out_json', help='Output coarse json path')
    args = parser.parse_args()
    generate_json(args.pred_dir, args.out_json)
