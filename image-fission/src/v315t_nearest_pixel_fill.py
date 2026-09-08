"""
v315t: scipy distance-transform nearest-pixel fill.

v315r/s issue: random sampling from extended camo source gives uniform
"horizontal band" texture — no spatial coherence.

v315t: use scipy.ndimage.distance_transform_edt to find the NEAREST non-text
pixel (in 2D) for each text pixel. Copy that pixel's color directly.
This gives perfect spatial coherence — fill matches local camo feature.

For the chain: it sits BETWEEN the text and the dog tag. We need the chain
preserved. So:
  - text_pix (gray<180) → fill
  - chain_pix → DON'T fill (preserve)
  - camo_pix (everything else) → source for fill

But chain is INSIDE the text band area. The nearest-camo logic must skip
chain pixels when finding the nearest source.
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
print(f'[v315t] Loaded {W}x{H}')

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

# Step 2: for each band, use scipy distance_transform_edt to find nearest camo pixel
# Source = non-text AND non-chain pixels
# Target = text pixels (gray<180)
# For each target pixel, find nearest source pixel in 2D

for band in BANDS:
    y0, y1 = band['y0'], band['y1']
    band_h = y1 - y0
    band_rgb = img_rgb[y0:y1, :].copy()
    band_gray = gray[y0:y1, :]
    chain_in_band = chain_mask[y0:y1, :] > 0

    # Build source/target masks
    text_pix = (band_gray < 180)  # text + antialiased edges
    source_pix = ~text_pix & ~chain_in_band  # camo pixels (NOT text, NOT chain)

    if text_pix.sum() == 0:
        print(f'[v315t] {band["name"]}: no text pixels')
        continue

    # distance_transform_edt with return_indices=True:
    # For each True pixel in input, find nearest False pixel
    # We want for each TEXT pixel, find nearest CAMO pixel
    # So input should have text=True, camo=False → distance_transform_edt gives
    # the location of the nearest False (camo) for each True (text)
    # But we need to ALSO exclude chain from being a source
    # Trick: set text=True, AND set chain=True (so chain is also "target" not "source")
    target = text_pix | chain_in_band
    # Now: True (text/chain) needs to find nearest False (camo)
    dist, (src_y, src_x) = distance_transform_edt(target, return_indices=True)

    # Apply: for each text pixel, copy camo color from (src_y, src_x)
    band_rgb[target] = band_rgb[src_y[target], src_x[target]]
    # Wait — this fills BOTH text and chain with camo. We want only text.
    # Let's redo: only fill text pixels
    band_rgb2 = img_rgb[y0:y1, :].copy()  # reset
    text_indices = np.where(text_pix)
    band_rgb2[text_indices] = band_rgb[src_y[text_pix], src_x[text_pix]]

    img_rgb[y0:y1, :] = band_rgb2
    print(f'[v315t] {band["name"]}: filled {int(text_pix.sum())} text px, max dist to nearest camo: {dist[text_pix].max():.1f}px')

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
    print(f'[v315t] {band["name"]}: "{text}" fsize={fsize} pos=({x},{y}) th={th}')

OUT = os.path.join(OUT_DIR, 'armed_text_only_v315t.jpg')
out_pil.save(OUT, quality=95)
print(f'[v315t] Saved {OUT}')

src_pil = Image.open(SRC).convert('RGB')
combined = Image.new('RGB', (src_pil.width * 2 + 30, src_pil.height), (50, 50, 50))
combined.paste(src_pil, (0, 0))
combined.paste(out_pil, (src_pil.width + 30, 0))
combined.save(os.path.join(OUT_DIR, 'armed_compare_v315t.jpg'), quality=90)
print('[v315t] Saved compare')
