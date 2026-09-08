"""
v315p: ONLY fill text pixels, leave surrounding camo untouched.

Bug in v315o: camo tile filled the ENTIRE band (y=462..765 for big1, etc.),
causing visible horizontal seams at the band boundaries because the clean
sample (y=100..400) has different lighting than the band y range.

Bug in v315n: NS inpaint with radius 12 + zeroed source produced diamond
artifacts.

v315p: tight mask = strict text pixels (gray<75) + small dilate (5px) - chain.
       TELEA radius 2 (very local, doesn't spread to non-text camo).
       Two passes: first removes dark text cores, second cleans antialiased
       edges with a looser mask.
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def load_jpg(p):
    with open(p, 'rb') as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

SRC = r'E:/Desktop/双接口/image-fission/jobs/v315/source_armed.jpg'
OUT_DIR = r'E:/Desktop/双接口/image-fission/jobs/v315'
FONT_PATH = r'C:/Windows/Fonts/impact.ttf'

img_bgr = load_jpg(SRC)
H, W = img_bgr.shape[:2]
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
print(f'[v315p] Loaded {W}x{H}')

# Original text y ranges (NOT band-inflated)
BANDS = [
    {'name': 'small', 'y0': 472, 'y1': 523, 'min_h': 25, 'min_w': 12, 'min_a': 200,  'target_h': 38},
    {'name': 'big1',  'y0': 541, 'y1': 640, 'min_h': 60, 'min_w': 30, 'min_a': 1000, 'target_h': 90},
    {'name': 'big2',  'y0': 656, 'y1': 755, 'min_h': 60, 'min_w': 30, 'min_a': 1000, 'target_h': 90},
]

# Step 1: chain mask (preserved through fill)
chain_mask = np.zeros((H, W), dtype=np.uint8)
for band in BANDS:
    y0, y1 = band['y0'], band['y1']
    bg = gray[y0:y1, :]
    dark = (bg < 75).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    for i in range(1, n):
        w, h, a = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        if not (w >= band['min_w'] and h >= band['min_h'] and a >= band['min_a']):
            x0 = stats[i, cv2.CC_STAT_LEFT]
            y_loc = stats[i, cv2.CC_STAT_TOP]
            cc = (labels[y_loc:y_loc+h, x0:x0+w] == i)
            chain_mask[y0+y_loc:y0+y_loc+h, x0:x0+w][cc] = 255

# Step 2: TIGHT mask = text pixels (gray<75) + 5px dilate - chain
# Note: only within the ORIGINAL text y range (not band-inflated)
mask1 = np.zeros((H, W), dtype=np.uint8)
for band in BANDS:
    y0, y1 = band['y0'], band['y1']
    bg = gray[y0:y1, :]
    text_pix = (bg < 75).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    d = cv2.dilate(text_pix, k, iterations=1)
    d[chain_mask[y0:y1, :] > 0] = 0
    mask1[y0:y1, :] = np.maximum(mask1[y0:y1, :], d)
print(f'[v315p] mask1: {int((mask1 > 0).sum())} px (tight text mask + 5px dilate)')

# Step 3: inpaint pass 1 — TELEA radius 3 (very local)
inp1 = cv2.inpaint(img_bgr, mask1, 3, cv2.INPAINT_TELEA)
img_rgb = cv2.cvtColor(inp1, cv2.COLOR_BGR2RGB).copy()
print(f'[v315p] Pass 1 done')

# Step 4: find remaining dark residue (gray<120 within original text bands)
# This catches anti-aliased edges that pass 1 didn't fully erase
residue_mask = np.zeros((H, W), dtype=np.uint8)
gray_after = cv2.cvtColor(inp1, cv2.COLOR_BGR2GRAY)
for band in BANDS:
    y0, y1 = band['y0'], band['y1']
    ga = gray_after[y0:y1, :]
    # Wider threshold to catch antialiased edges
    resid = (ga < 120).astype(np.uint8) * 255
    # Subtract chain
    resid[chain_mask[y0:y1, :] > 0] = 0
    # Tiny dilate
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    resid_d = cv2.dilate(resid, k, iterations=1)
    # But also need to subtract the original text region (so we don't keep
    # painting new text over old residue that's already removed)
    residue_mask[y0:y1, :] = np.maximum(residue_mask[y0:y1, :], resid_d)
# Subtract chain one more time
residue_mask[chain_mask > 0] = 0
print(f'[v315p] residue_mask: {int((residue_mask > 0).sum())} px')

# Step 5: inpaint pass 2 on residue (TELEA radius 2)
if residue_mask.sum() > 0:
    inp2 = cv2.inpaint(inp1, residue_mask, 2, cv2.INPAINT_TELEA)
    img_rgb = cv2.cvtColor(inp2, cv2.COLOR_BGR2RGB).copy()
    print(f'[v315p] Pass 2 done')

# Step 6: write new text — at EXACT same y ranges as original
out_pil = Image.fromarray(img_rgb)
draw = ImageDraw.Draw(out_pil)

def get_font(target_h, font_path, sample='ABCDEFG'):
    for size in range(30, 200):
        font = ImageFont.truetype(font_path, size)
        bbox = draw.textbbox((0, 0), sample, font=font)
        h = bbox[3] - bbox[1]
        if h >= target_h:
            return font, size
    return ImageFont.truetype(font_path, 200), 200

NEW_TEXTS = {
    'small': 'HONOR OUR VETERANS',
    'big1':  'BRAVE',
    'big2':  'LEGION',
}

def orig_x_extent(gray, y0, y1, min_w, min_h, min_a):
    bg = gray[y0:y1, :]
    dark = (bg < 75).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    xs = []
    for i in range(1, n):
        w, h, a = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        if w >= min_w and h >= min_h and a >= min_a:
            xs.append(stats[i, cv2.CC_STAT_LEFT])
            xs.append(stats[i, cv2.CC_STAT_LEFT] + w)
    return (min(xs), max(xs)) if xs else (0, 0)

for band in BANDS:
    text = NEW_TEXTS[band['name']]
    font, fsize = get_font(band['target_h'], FONT_PATH)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    # Use ORIGINAL text y range (not feathered)
    ox0, ox1 = orig_x_extent(gray, band['y0'], band['y1'],
                             band['min_w'], band['min_h'], band['min_a'])
    cx = (ox0 + ox1) // 2
    cy = (band['y0'] + band['y1']) // 2
    x = cx - tw // 2 - bbox[0]
    y = cy - th // 2 - bbox[1]
    draw.text((x, y), text, font=font, fill=(20, 20, 20))
    print(f'[v315p] {band["name"]}: "{text}" fsize={fsize} pos=({x},{y}) th={th} cx={cx}')

OUT = os.path.join(OUT_DIR, 'armed_text_only_v315p.jpg')
out_pil.save(OUT, quality=95)
print(f'[v315p] Saved {OUT}')

src_pil = Image.open(SRC).convert('RGB')
combined = Image.new('RGB', (src_pil.width * 2 + 30, src_pil.height), (50, 50, 50))
combined.paste(src_pil, (0, 0))
combined.paste(out_pil, (src_pil.width + 30, 0))
combined.save(os.path.join(OUT_DIR, 'armed_compare_v315p.jpg'), quality=90)
print('[v315p] Saved compare')
