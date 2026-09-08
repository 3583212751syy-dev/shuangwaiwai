"""
v315o: per-band camo tile fill with feathered edge blending.

Bug in v315n: cv2.inpaint NS radius=12 + zeroed source + huge mask (365k px)
produced diamond-shaped Laplace equation artifacts.

v315o strategy: don't use inpaint at all.
  1. Sample camo from a KNOWN CLEAN region: y=100..400 (above all text)
  2. Tile (resize) it to each band
  3. Feather top/bottom 20px to blend with surrounding camo
  4. Subtract chain (chain pixels untouched)
  5. Write new text on top — slightly smaller fonts for breathing room
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
print(f'[v315o] Loaded {W}x{H}')

# Band y ranges (text+padding)
BANDS = [
    {'name': 'small', 'y0': 462, 'y1': 533, 'min_h': 25, 'min_w': 12, 'min_a': 200,  'target_h': 38, 'feather': 20},
    {'name': 'big1',  'y0': 531, 'y1': 650, 'min_h': 60, 'min_w': 30, 'min_a': 1000, 'target_h': 90, 'feather': 20},
    {'name': 'big2',  'y0': 646, 'y1': 765, 'min_h': 60, 'min_w': 30, 'min_a': 1000, 'target_h': 90, 'feather': 20},
]

# Step 1: classify chain CCs (preserved through fill)
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

# Step 2: sample clean camo from y=100..400 (above all text)
CLEAN_Y0, CLEAN_Y1 = 100, 400
camo_sample = img_rgb[CLEAN_Y0:CLEAN_Y1, :, :].astype(np.float32)
print(f'[v315o] camo sample: {camo_sample.shape}')

# Step 3: per-band tile + feather
for band in BANDS:
    y0, y1 = band['y0'], band['y1']
    band_h = y1 - y0

    # Resize clean camo to (band_h, W)
    camo_tile = cv2.resize(camo_sample, (W, band_h), interpolation=cv2.INTER_LINEAR)

    # Build mask: float32, top/bottom feathered, 0 at chain
    mask = np.ones((band_h, W), dtype=np.float32)
    fe = band['feather']
    # Top feather: 0 → 1
    mask[:fe, :] *= np.linspace(0, 1, fe, dtype=np.float32)[:, None]
    # Bottom feather: 1 → 0
    mask[-fe:, :] *= np.linspace(1, 0, fe, dtype=np.float32)[:, None]
    # Chain always 0
    mask[chain_mask[y0:y1, :] > 0] = 0.0

    # Blend: out = camo_tile * mask + orig * (1 - mask)
    orig = img_rgb[y0:y1, :].astype(np.float32)
    blended = camo_tile * mask[:, :, None] + orig * (1.0 - mask[:, :, None])
    img_rgb[y0:y1, :] = np.clip(blended, 0, 255).astype(np.uint8)
    print(f'[v315o] {band["name"]}: blended {band_h}px band, chain-pixels preserved')

# Step 4: write new text — smaller fonts for breathing room
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
    'small': 'HONOR THOSE WHO SERVED',
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
    # Find original text x-center, but use original text y-band (not feathered)
    inner_y0 = band['y0'] + band['feather']
    inner_y1 = band['y1'] - band['feather']
    ox0, ox1 = orig_x_extent(gray, inner_y0, inner_y1,
                             band['min_w'], band['min_h'], band['min_a'])
    cx = (ox0 + ox1) // 2
    cy = (inner_y0 + inner_y1) // 2
    x = cx - tw // 2 - bbox[0]
    y = cy - th // 2 - bbox[1]
    draw.text((x, y), text, font=font, fill=(20, 20, 20))
    print(f'[v315o] {band["name"]}: "{text}" fsize={fsize} pos=({x},{y}) th={th} cx={cx}')

OUT = os.path.join(OUT_DIR, 'armed_text_only_v315o.jpg')
out_pil.save(OUT, quality=95)
print(f'[v315o] Saved {OUT}')

src_pil = Image.open(SRC).convert('RGB')
combined = Image.new('RGB', (src_pil.width * 2 + 30, src_pil.height), (50, 50, 50))
combined.paste(src_pil, (0, 0))
combined.paste(out_pil, (src_pil.width + 30, 0))
combined.save(os.path.join(OUT_DIR, 'armed_compare_v315o.jpg'), quality=90)
print('[v315o] Saved compare')
