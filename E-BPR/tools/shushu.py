from glob import glob
from collections import defaultdict
import json
import os

detail_files = glob('maskrcnn_r50/patches/detail_dir/train/*.json')
img_patch_count = defaultdict(int)

for f in detail_files:
    with open(f, 'r') as fp:
        info = json.load(fp)
    img_patch_count[info['image_name']] += len(info['patches'])

values = list(img_patch_count.values())
print(f"Images: {len(values)}")
print(f"Total patches: {sum(values)}")
print(f"Avg patches per image: {sum(values)/len(values):.2f}")
