"""诊断：导出 b78e60 中间产物（crop / mask / lama 擦除结果）定位为何旧字未被擦除。"""
import json, sys
from pathlib import Path
from PIL import Image
import numpy as np
sys.path.insert(0, '.')
sys.path.insert(0, 'src')
from scipy import ndimage
from skimage import morphology
import v268_lama_clean as lc

cfg = json.load(open('regression_set/set.json', encoding='utf-8'))
ic = [i for i in cfg['images'] if i['id'] == 'b78e60'][0]
img = Image.open(ic['path']).convert('RGB')
W, H = img.size
it = ic['text_plan'][1]  # "STEEL" 中间行
x1, y1, x2, y2 = [int(v) for v in it['bbox']]
print("bbox", x1, y1, x2, y2, "img", W, H)
margin = 200
cx1, cy1, cx2, cy2 = max(0, x1 - margin), max(0, y1 - margin), min(W, x2 + margin), min(H, y2 + margin)
crop = img.crop((cx1, cy1, cx2, cy2))
bx1, by1, bx2, by2 = x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1
mask_arr = np.zeros((crop.height, crop.width), np.uint8)
pad_x = max(4, int((bx2 - bx1) * 0.10)); pad_y = max(4, int((by2 - by1) * 0.10))
mx1, my1 = max(0, bx1 - pad_x), max(0, by1 - pad_y)
mx2, my2 = min(crop.width, bx2 + pad_x), min(crop.height, by2 + pad_y)
mask_arr[my1:my2, mx1:mx2] = 255
selem = morphology.disk(10)
mask_arr = ndimage.binary_dilation(mask_arr > 0, structure=selem).astype(np.uint8) * 255
mask_pil = Image.fromarray(mask_arr, 'L')
crop.save('jobs/_diag_crop.png'); mask_pil.save('jobs/_diag_mask.png')
cleaned = lc.lama_inpaint(crop, mask_pil, removal_strength=235, edge_smoothness=4)
cleaned.save('jobs/_diag_cleaned.png')
print("crop size", crop.size, "mask frac", round((mask_arr > 0).mean(), 3))
print("done")
