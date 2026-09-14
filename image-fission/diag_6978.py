"""诊断 6978 顶部弧字：导出 crop/mask/cleaned，看弧字是否被擦除。"""
import json, sys
from PIL import Image
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'src')
from scipy import ndimage
from skimage import morphology
import v268_lama_clean as lc

cfg = json.load(open('regression_set/set.json', encoding='utf-8'))
ic = [i for i in cfg['images'] if i['id'] == '6978'][0]
img = Image.open(ic['path']).convert('RGB')
W, H = img.size
it = ic['text_plan'][0]  # arc
x1, y1, x2, y2 = [int(v) for v in it['bbox']]
print("arc bbox", x1, y1, x2, y2, "img", W, H, "margin", it.get("margin"), "dilate", it.get("dilate"))
margin = int(it.get("margin", 180))
cx1, cy1, cx2, cy2 = max(0, x1 - margin), max(0, y1 - margin), min(W, x2 + margin), min(H, y2 + margin)
crop = img.crop((cx1, cy1, cx2, cy2))
bx1, by1, bx2, by2 = x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1
mask_arr = np.zeros((crop.height, crop.width), np.uint8)
pad_x = max(4, int((bx2 - bx1) * 0.10)); pad_y = max(4, int((by2 - by1) * 0.10))
mx1, my1 = max(0, bx1 - pad_x), max(0, by1 - pad_y)
mx2, my2 = min(crop.width, bx2 + pad_x), min(crop.height, by2 + pad_y)
mask_arr[my1:my2, mx1:mx2] = 255
selem = morphology.disk(max(int(it.get("dilate", 36)), 8))
mask_arr = ndimage.binary_dilation(mask_arr > 0, structure=selem).astype(np.uint8) * 255
mask_pil = Image.fromarray(mask_arr, 'L')
crop.save('jobs/_d6978_crop.png'); mask_pil.save('jobs/_d6978_mask.png')
cleaned = lc.lama_inpaint(crop, mask_pil, removal_strength=int(it.get("removal_strength", 250)),
                          edge_smoothness=int(it.get("edge_smoothness", 3)))
cleaned.save('jobs/_d6978_cleaned.png')
print("crop", crop.size, "mask frac", round((mask_arr > 0).mean(), 3))
