"""
v315v: tight mask (gray<100) + scipy nearest-camo fill (with cap).

v315q issues:
  - gray<100 misses anti-aliased edges → faint ghost shadows
  - median fill = flat patch (no spatial coherence)

v315t issues:
  - gray<180 misclassifies dark camo as text → nearest camo = 451px away
    in bright region → bright white patch

v315v compromise:
  - TIGHT mask: gray<100 (only true text cores) + dilate 4px (catch
    immediate anti-alias halo but not misclassify camo)
  - For each text pixel, use distance_transform_edt to find NEAREST camo
    pixel in 2D (chain pixels also excluded from being source)
  - CAP distance at 30px — if nearest camo is > 30px away, use band median
    (prevents 451px jumps into wrong lighting regions)
  - This gives spatial coherence within 30px, falls back to flat median
    for the rare text pixel > 30px from camo (usually a letter inside
    the band, surrounded by text, very rare)
"""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
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
print(f'[v315v] Loaded {W}x{H}')

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

# Step 2: tight mask = gray<100, dilate 4px (catch immediate anti-alias halo)
DIST_CAP = 30  # pixels

for band in BANDS:
    y0, y1 = band['y0'], band['y1']
    band_h = y1 - y0
    band_rgb = img_rgb[y0:y1, :].copy()
    band_gray = gray[y0:y1, :]
    chain_in_band = chain_mask[y0:y1, :] > 0

    # Tight text mask + 4px dilate
    text_pix = (band_gray < 100)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    text_pix_d = cv2.dilate(text_pix.astype(np.uint8) * 255, k, iterations=1) > 0
    # Subtract chain from text mask
    text_pix_d = text_pix_d & ~chain_in_band

    if text_pix_d.sum() == 0:
        print(f'[v315v] {band["name"]}: no text pixels')
        continue

    # Camo source = non-text, non-chain
    camo_pix = ~text_pix_d & ~chain_in_band
    camo_pixels = band_rgb[camo_pix]
    if len(camo_pixels) == 0:
        continue
    median_color = np.median(camo_pixels, axis=0)

    # distance_transform_edt
    # target = text (we want to fill these)
    # source = camo (we look up these)
    # chain must NOT be a source — mark as "target" too
    target = text_pix_d | chain_in_band
    dist, (src_y, src_x) = distance_transform_edt(target, return_indices=True)

    # For each text pixel, if dist <= cap, use nearest camo color
    text_indices = np.where(text_pix_d)
    n_text = len(text_indices[0])
    if n_text == 0:
        continue

    # Get distance for each text pixel
    text_dist = dist[text_pix_d]
    # Cap: pixels within DIST_CAP use nearest camo; beyond use median
    use_nearest = text_dist <= DIST_CAP
    use_median = ~use_nearest

    # Fill
    fill_colors = np.zeros((n_text, 3), dtype=np.uint8)
    fill_colors[use_nearest] = band_rgb[src_y[text_pix_d][use_nearest], src_x[text_pix_d][use_nearest]]
    fill_colors[use_median] = median_color

    band_rgb[text_indices] = fill_colors
    img_rgb[y0:y1, :] = band_rgb
    n_capped = int(use_median.sum())
    print(f'[v315v] {band["name"]}: filled {n_text} text px, {n_capped} capped to median (dist>{DIST_CAP}px), max dist={float(text_dist.max()):.1f}px')

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
    print(f'[v315v] {band["name"]}: "{text}" fsize={fsize} pos=({x},{y}) th={th}')

OUT = os.path.join(OUT_DIR, 'armed_text_only_v315v.jpg')
out_pil.save(OUT, quality=95)
print(f'[v315v] Saved {OUT}')

src_pil = Image.open(SRC).convert('RGB')
combined = Image.new('RGB', (src_pil.width * 2 + 30, src_pil.height), (50, 50, 50))
combined.paste(src_pil, (0, 0))
combined.paste(out_pil, (src_pil.width + 30, 0))
combined.save(os.path.join(OUT_DIR, 'armed_compare_v315v.jpg'), quality=90)
print('[v315v] Saved compare')
