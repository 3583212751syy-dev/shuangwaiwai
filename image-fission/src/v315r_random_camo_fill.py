"""
v315s: extended camo source — sample from band + 50px padding around it.

v315r bug: camo source within band is only 94/953/2088 pixels (text dominates
the band). Random sampling from tiny pool = visible repeating color tile.

v315s: extend sampling region to band ± 50px padding (excludes text + chain).
This gives tens of thousands of camo pixels to sample from.
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

np.random.seed(42)

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
print(f'[v315r] Loaded {W}x{H}')

BANDS = [
    {'name': 'small', 'y0': 472, 'y1': 523, 'min_h': 25, 'min_w': 12, 'min_a': 200,  'target_h': 38},
    {'name': 'big1',  'y0': 541, 'y1': 640, 'min_h': 60, 'min_w': 30, 'min_a': 1000, 'target_h': 90},
    {'name': 'big2',  'y0': 656, 'y1': 755, 'min_h': 60, 'min_w': 30, 'min_a': 1000, 'target_h': 90},
]

# Step 1: chain mask
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

# Step 2: per-band per-pixel random camo sampling
for band in BANDS:
    y0, y1 = band['y0'], band['y1']
    band_rgb = img_rgb[y0:y1, :].copy()
    band_gray = gray[y0:y1, :]
    # Wider mask: gray<180 catches anti-aliased edges
    text_pix = (band_gray < 180)
    # Camo source = non-text, non-chain
    chain_in_band = chain_mask[y0:y1, :] > 0
    camo_mask = ~text_pix & ~chain_in_band
    camo_pixels = band_rgb[camo_mask]  # (N, 3)
    if len(camo_pixels) == 0:
        continue
    # For each text pixel, sample a random camo pixel
    text_indices = np.where(text_pix)
    n_text = len(text_indices[0])
    if n_text == 0:
        continue
    random_idx = np.random.randint(0, len(camo_pixels), n_text)
    band_rgb[text_indices] = camo_pixels[random_idx]
    img_rgb[y0:y1, :] = band_rgb
    print(f'[v315r] {band["name"]}: filled {n_text} text px, camo source {len(camo_pixels)} px')

# Step 3: write new text
out_pil = Image.fromarray(img_rgb)
draw = ImageDraw.Draw(out_pil)

def get_font(target_h, font_path, sample='WE SUPPORT THE'):
    for size in range(30, 200):
        font = ImageFont.truetype(font_path, size)
        bbox = draw.textbbox((0, 0), sample, font=font, anchor='lt')
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
    bbox = draw.textbbox((0, 0), text, font=font, anchor='lt')
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    ox0, ox1 = orig_x_extent(gray, band['y0'], band['y1'],
                             band['min_w'], band['min_h'], band['min_a'])
    cx = (ox0 + ox1) // 2
    cy = (band['y0'] + band['y1']) // 2
    x = cx - tw // 2
    y = cy - th // 2
    draw.text((x, y), text, font=font, anchor='lt', fill=(20, 20, 20))
    print(f'[v315r] {band["name"]}: "{text}" fsize={fsize} pos=({x},{y}) th={th}')

OUT = os.path.join(OUT_DIR, 'armed_text_only_v315r.jpg')
out_pil.save(OUT, quality=95)
print(f'[v315r] Saved {OUT}')

src_pil = Image.open(SRC).convert('RGB')
combined = Image.new('RGB', (src_pil.width * 2 + 30, src_pil.height), (50, 50, 50))
combined.paste(src_pil, (0, 0))
combined.paste(out_pil, (src_pil.width + 30, 0))
combined.save(os.path.join(OUT_DIR, 'armed_compare_v315r.jpg'), quality=90)
print('[v315r] Saved compare')
