"""v319j: Fix v319i's mask bug. Use v316e's proven strict classification.

v319i bug: is_letter = h >= band_h * 0.35 — too lax, dark camo patches with
height >= 18px got classified as letter, so text_core mask covered almost
the entire image. LaMa "rebuilt" the whole image as gray camo + hallucinated
the original text back.

Fix: letter = h >= band_h * 0.6 (only true vertical letter strokes)
     serif  = w >= 30, h >= 20, a >= 200, h < band_h * 0.5 (crossbars)
     chain  = rest of small dark CCs

This is the same logic as v316e, which produced correct masks.
"""
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path("E:/Desktop/双接口/image-fission")
SRC = str(PROJECT / "jobs" / "v315" / "source_armed.jpg")
OUT_DIR = str(PROJECT / "jobs" / "v319j")
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

# Strict classification (v316e proven)
chain_mask = np.zeros((H, W), dtype=np.uint8)
letter_mask = np.zeros((H, W), dtype=bool)
serif_mask = np.zeros((H, W), dtype=bool)

for band in BANDS_DEF:
    y0, y1 = band['y0'], band['y1']
    band_h = y1 - y0
    band_gray = gray[y0:y1, :]
    dark = (band_gray < 80).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    for i in range(1, n):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        a = stats[i, cv2.CC_STAT_AREA]
        # Letter = vertical stroke, full band height
        is_letter = (h >= band_h * 0.6) and (a >= 1000)
        # Serif = horizontal crossbar (w >= 30, h between 8 and 50% band_h, area big)
        is_serif = (not is_letter) and (w >= 30) and (h >= 8) and (h < band_h * 0.5) and (a >= 200)
        x0 = stats[i, cv2.CC_STAT_LEFT]
        y_loc = stats[i, cv2.CC_STAT_TOP]
        if is_letter:
            letter_mask[y0+y_loc:y0+y_loc+h, x0:x0+w] = True
        elif is_serif:
            serif_mask[y0+y_loc:y0+y_loc+h, x0:x0+w] = True
        else:
            cc = (labels[y_loc:y_loc+h, x0:x0+w] == i)
            chain_mask[y0+y_loc:y0+y_loc+h, x0:x0+w][cc] = 255

# Text mask = letter + serif (both are parts of the text characters)
text_mask = (letter_mask | serif_mask) & (chain_mask == 0)
print(f'[v319j] letter px: {int(letter_mask.sum())}')
print(f'[v319j] serif px: {int(serif_mask.sum())}')
print(f'[v319j] text mask px: {int(text_mask.sum())}')
print(f'[v319j] chain mask px: {int((chain_mask > 0).sum())}')

# Per band stats
for band in BANDS_DEF:
    y0, y1 = band['y0'], band['y1']
    band_text = int(text_mask[y0:y1, :].sum())
    band_chain = int((chain_mask[y0:y1, :] > 0).sum())
    band_area = (y1 - y0) * W
    print(f'  {band["name"]}: text={band_text} ({band_text/band_area*100:.1f}%), chain={band_chain}')

# Visualize
mask_vis = np.zeros((H, W, 3), dtype=np.uint8)
mask_vis[letter_mask] = [255, 0, 0]  # red = letter
mask_vis[serif_mask] = [255, 165, 0]  # orange = serif
mask_vis[chain_mask > 0] = [0, 255, 0]  # green = chain
overlay = (img_rgb * 0.5 + mask_vis * 0.5).astype(np.uint8)
Image.fromarray(overlay).save(str(Path(OUT_DIR) / 'mask_overlay_full.png'))

crop = overlay[440:780, :, :]
Image.fromarray(crop).save(str(Path(OUT_DIR) / 'mask_overlay_crop.png'))
print(f'[v319j] Saved mask visualizations to {OUT_DIR}/mask_overlay_*.png')
