"""
v316b: ARMED text-only fission — fix v316 issues.

v316 problems:
  1. Small band "WE HONOR THE BRAVE" too long → jammed
  2. Horizontal seams at band edges (camo + median blending visible)
  3. ARMED font's horizontal serifs were deleted (mistaken for decoration)
  4. Background patch from median fallback still visible

v316b fixes:
  1. Shorten small text: "HONOR THE BRAVE" (15 chars)
  2. DECREASE DIST_CAP to 8px — most text pixels have camo within 8px
     (letter strokes are 30-50px wide, gap between strokes ~10-20px)
  3. Identify "horizontal serif" CCs separately: w > 200 AND h < 30,
     keep them as part of text area BUT exclude from camo source so
     they don't influence fill colors
  4. Use full-image scipy fill instead of per-band (smoother transitions)

Actually, simpler approach: identify text vs non-text by:
  - text CC: h >= 70% band height (taller than half-band)
  - everything else: chain/decoration/serif, EXCLUDE from text mask
    BUT preserve in image (don't fill)

The horizontal serifs in ARMED font are actually PART of the letters.
They span band edges (e.g. ARMED's horizontal in 'A', 'E'). When I delete
text and try to fill, the serif line gets deleted and replaced with
camo/median → visible seam.

Fix: identify text as FULL-HEIGHT strokes only (h >= 70% band_h).
Horizontal serifs are lower height, will be treated as part of fill
source (chain mask). They'll show through the camo fill but new text
will cover them.

Wait, that doesn't work — the serif is BLACK, will show as black line.

Better fix: identify text strictly as full-height CCs only. After
filling, the searif line stays black. New Anton font will OVERWRITE
those black lines (Anton has horizontal strokes too).

Let me also reduce DIST_CAP to 8 — tighter fill, more median fallback
but median color is uniform per band so less visible seams.
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
ANTON_PATH = r'E:/Desktop/双接口/image-fission/ComfyUI/models/fonts/Anton-Regular.ttf'

img_bgr = load_jpg(SRC)
H, W = img_bgr.shape[:2]
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
print(f'[v316b] Loaded {W}x{H}')

BANDS = [
    {'name': 'small', 'y0': 472, 'y1': 523, 'target_h': 51, 'font_size': 58},
    {'name': 'big1',  'y0': 541, 'y1': 640, 'target_h': 99, 'font_size': 115},
    {'name': 'big2',  'y0': 656, 'y1': 755, 'target_h': 99, 'font_size': 115},
]

# Shorter text for small band
NEW_TEXTS = {
    'small': 'HONOR OUR VETERANS',
    'big1':  'VETERANS',
    'big2':  'LEGION',
}

FONTS = {}
for band in BANDS:
    FONTS[band['name']] = ImageFont.truetype(ANTON_PATH, band['font_size'])

# Step 1: chain/serif mask = ALL CCs that are NOT full-height text
chain_mask = np.zeros((H, W), dtype=np.uint8)
for band in BANDS:
    y0, y1 = band['y0'], band['y1']
    band_h = y1 - y0
    band_gray = gray[y0:y1, :]
    dark = (band_gray < 80).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    for i in range(1, n):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        a = stats[i, cv2.CC_STAT_AREA]
        # Tighter: text = full band height CC
        # ARMED letters span h=99 in 99-px band → ratio=1.0
        # WE SUPPORT THE letters span h=51 in 51-px band → ratio=1.0
        is_text = (h >= band_h * 0.7) and (a >= 800)
        if not is_text:
            x0 = stats[i, cv2.CC_STAT_LEFT]
            y_loc = stats[i, cv2.CC_STAT_TOP]
            cc = (labels[y_loc:y_loc+h, x0:x0+w] == i)
            chain_mask[y0+y_loc:y0+y_loc+h, x0:x0+w][cc] = 255

# Step 2: Fill text bands
DIST_CAP = 12  # tight cap (most text pixels have camo within 12px)
for band in BANDS:
    y0, y1 = band['y0'], band['y1']
    band_rgb = img_rgb[y0:y1, :].copy()
    band_gray = gray[y0:y1, :]
    chain_in_band = chain_mask[y0:y1, :] > 0

    text_pix = (band_gray < 100)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    text_pix_d = cv2.dilate(text_pix.astype(np.uint8) * 255, k, iterations=1) > 0
    text_pix_d = text_pix_d & ~chain_in_band

    if text_pix_d.sum() == 0:
        print(f'[v316b] {band["name"]}: no text pixels, skipping')
        continue

    # Camo source: pixels that are NOT text, NOT chain
    camo_pix = ~text_pix_d & ~chain_in_band
    camo_pixels = band_rgb[camo_pix]
    if len(camo_pixels) == 0:
        continue
    median_color = np.median(camo_pixels, axis=0)

    target = text_pix_d | chain_in_band
    dist, (src_y, src_x) = distance_transform_edt(target, return_indices=True)

    text_indices = np.where(text_pix_d)
    n_text = len(text_indices[0])
    text_dist = dist[text_pix_d]
    use_nearest = text_dist <= DIST_CAP
    use_median = ~use_nearest

    fill_colors = np.zeros((n_text, 3), dtype=np.uint8)
    fill_colors[use_nearest] = band_rgb[src_y[text_pix_d][use_nearest], src_x[text_pix_d][use_nearest]]
    fill_colors[use_median] = median_color

    band_rgb[text_indices] = fill_colors
    img_rgb[y0:y1, :] = band_rgb
    print(f'[v316b] {band["name"]}: filled {n_text} text px, {int(use_median.sum())} capped (>{DIST_CAP}px)')

# Step 3: Write text
out_pil = Image.fromarray(img_rgb)
draw = ImageDraw.Draw(out_pil)

def orig_x_extent(gray, y0, y1, band_h):
    bg = gray[y0:y1, :]
    dark = (bg < 80).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    xs = []
    for i in range(1, n):
        h = stats[i, cv2.CC_STAT_HEIGHT]
        a = stats[i, cv2.CC_STAT_AREA]
        if h >= band_h * 0.7 and a >= 800:
            xs.append(stats[i, cv2.CC_STAT_LEFT])
            xs.append(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH])
    return (min(xs), max(xs)) if xs else (0, W)

for band in BANDS:
    text = NEW_TEXTS[band['name']]
    font = FONTS[band['name']]
    bbox = draw.textbbox((0, 0), text, font=font, anchor='lt')
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    ox0, ox1 = orig_x_extent(gray, band['y0'], band['y1'], band['y1'] - band['y0'])
    cx = (ox0 + ox1) // 2
    cy = (band['y0'] + band['y1']) // 2
    x = cx - tw // 2 - bbox[0]
    y = cy - th // 2 - bbox[1]
    draw.text((x, y), text, font=font, anchor='lt', fill=(20, 20, 20))
    print(f'[v316b] {band["name"]}: "{text}" pos=({x},{y})')

OUT = os.path.join(OUT_DIR, 'armed_text_only_v316b.jpg')
out_pil.save(OUT, quality=95)

src_pil = Image.open(SRC).convert('RGB')
combined = Image.new('RGB', (src_pil.width * 2 + 30, src_pil.height), (50, 50, 50))
combined.paste(src_pil, (0, 0))
combined.paste(out_pil, (src_pil.width + 30, 0))
combined.save(os.path.join(OUT_DIR, 'armed_compare_v316b.jpg'), quality=90)
print(f'[v316b] Saved {OUT}')
