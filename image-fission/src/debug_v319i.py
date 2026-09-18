"""Debug v319i: visualize each mask to find the bug."""
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

PROJECT = Path("E:/Desktop/双接口/image-fission")
SRC = str(PROJECT / "jobs" / "v315" / "source_armed.jpg")
OUT_DIR = str(PROJECT / "jobs" / "v319i")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

def load_jpg(p):
    with open(p, 'rb') as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

img_bgr = load_jpg(SRC)
H, W = img_bgr.shape[:2]
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

BANDS_DEF = [
    {'name': 'small', 'y0': 472, 'y1': 523},
    {'name': 'big1',  'y0': 541, 'y1': 640},
    {'name': 'big2',  'y0': 656, 'y1': 755},
]

chain_mask = np.zeros((H, W), dtype=np.uint8)
letter_mask = np.zeros((H, W), dtype=bool)
for band in BANDS_DEF:
    y0, y1 = band['y0'], band['y1']
    band_h = y1 - y0
    band_gray = gray[y0:y1, :]
    dark = (band_gray < 80).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    for i in range(1, n):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        is_letter = (h >= band_h * 0.35)
        x0 = stats[i, cv2.CC_STAT_LEFT]
        y_loc = stats[i, cv2.CC_STAT_TOP]
        if is_letter:
            letter_mask[y0+y_loc:y0+y_loc+h, x0:x0+w] = True
        else:
            cc = (labels[y_loc:y_loc+h, x0:x0+w] == i)
            chain_mask[y0+y_loc:y0+y_loc+h, x0:x0+w][cc] = 255

text_core = letter_mask & (gray < 80) & (chain_mask == 0)
print(f'text_core px: {int(text_core.sum())}')

aa_ring = letter_mask & (gray >= 80) & (gray < 160) & (chain_mask == 0)
aa_halo = (~letter_mask) & (gray >= 80) & (gray < 160)
core_dilated = cv2.dilate(text_core.astype(np.uint8) * 255,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                          iterations=1) > 0
aa_mask = aa_ring | (aa_halo & core_dilated)
aa_mask = aa_mask & (~text_core)
print(f'aa_ring (letter & 80-160) px: {int(aa_ring.sum())}')
print(f'aa_halo (NOT letter & 80-160) px: {int(aa_halo.sum())}')
print(f'core_dilated px: {int(core_dilated.sum())}')
print(f'aa_halo & core_dilated px: {int((aa_halo & core_dilated).sum())}')
print(f'final aa_mask px: {int(aa_mask.sum())}')

# Visualize each mask on top of source
def make_overlay(mask_bool, color):
    overlay = img_rgb.copy()
    overlay[mask_bool] = (overlay[mask_bool] * 0.3 + np.array(color) * 0.7).astype(np.uint8)
    return overlay

Image.fromarray(make_overlay(text_core, [255, 0, 0])).save(str(Path(OUT_DIR) / 'mask_text_core.png'))
Image.fromarray(make_overlay(aa_ring, [0, 255, 0])).save(str(Path(OUT_DIR) / 'mask_aa_ring.png'))
Image.fromarray(make_overlay(aa_halo, [0, 0, 255])).save(str(Path(OUT_DIR) / 'mask_aa_halo.png'))
Image.fromarray(make_overlay(core_dilated, [255, 255, 0])).save(str(Path(OUT_DIR) / 'mask_core_dilated.png'))
Image.fromarray(make_overlay(aa_mask, [255, 0, 255])).save(str(Path(OUT_DIR) / 'mask_aa_mask_final.png'))
print('Saved 5 mask visualizations')
