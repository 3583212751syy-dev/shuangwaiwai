"""
v315q: skip cv2.inpaint entirely. Use per-band median+noise fill.

After 12+ iterations of inpaint attempts, all cv2 inpaint variants fail on
high-frequency camo:
  - NS radius 12: diamond artifacts (Laplace equation solution)
  - TELEA radius 3: bright spots (FMM uses far-context pixels)
  - TELEA radius 2: residue blobs
  - Both with zeroed source: text leaking through
  - Both with non-zeroed source: text leaking through
  - Both with band-wide mask: visible seam at band boundary
  - Both with tight mask: residue around letter edges

v315q strategy: per-band, fill text pixels with median camo color of the
BAND ITSELF (not above/below, which is contaminated) + small Gaussian
noise matching the camo stddev. No inpaint.

Text positioning: use PIL anchor='lt' (left-top) so (x, y) is the
top-left of the bounding box. Position y = band center - text height / 2.
Font size tuned to give th = band_h * 0.85.
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
print(f'[v315q] Loaded {W}x{H}')

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

# Step 2: per-band fill — median color of band + Gaussian noise
for band in BANDS:
    y0, y1 = band['y0'], band['y1']
    band_rgb = img_rgb[y0:y1, :]
    band_gray = gray[y0:y1, :]
    # Text pixels: gray<100 to catch anti-aliased edges
    text_pix = (band_gray < 100)
    # Camo source = non-text, non-chain pixels in this band
    chain_in_band = chain_mask[y0:y1, :] > 0
    camo_mask = ~text_pix & ~chain_in_band
    camo_pixels = band_rgb[camo_mask]
    if len(camo_pixels) == 0:
        continue
    median_color = np.median(camo_pixels, axis=0)  # (3,)
    stddev = np.std(camo_pixels, axis=0)  # (3,)
    # Per-pixel noise: gaussian with stddev scaled to 0.3x for subtle texture
    noise_std = stddev * 0.3
    # Fill text pixels: median + noise
    text_indices = np.where(text_pix)
    n_text = len(text_indices[0])
    if n_text == 0:
        continue
    noise = np.random.normal(0, noise_std, (n_text, 3))
    fill_colors = median_color[None, :] + noise
    fill_colors = np.clip(fill_colors, 0, 255).astype(np.uint8)
    band_rgb[text_indices] = fill_colors
    img_rgb[y0:y1, :] = band_rgb
    print(f'[v315q] {band["name"]}: filled {n_text} text pixels, median=({int(median_color[0])},{int(median_color[1])},{int(median_color[2])}), stddev=({int(stddev[0])},{int(stddev[1])},{int(stddev[2])})')

# Step 3: write new text using anchor='lt' for clean positioning
out_pil = Image.fromarray(img_rgb)
draw = ImageDraw.Draw(out_pil)

def get_font_for_anchor(target_h, font_path, sample='WE SUPPORT THE'):
    """Find font size that gives bbox height matching target_h using anchor='lt'."""
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
    font, fsize = get_font_for_anchor(band['target_h'], FONT_PATH)
    bbox = draw.textbbox((0, 0), text, font=font, anchor='lt')
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    # Center on original text x
    ox0, ox1 = orig_x_extent(gray, band['y0'], band['y1'],
                             band['min_w'], band['min_h'], band['min_a'])
    cx = (ox0 + ox1) // 2
    cy = (band['y0'] + band['y1']) // 2
    # With anchor='lt', (x, y) is the top-left of bbox
    # So to center: x = cx - tw//2, y = cy - th//2
    x = cx - tw // 2
    y = cy - th // 2
    draw.text((x, y), text, font=font, anchor='lt', fill=(20, 20, 20))
    print(f'[v315q] {band["name"]}: "{text}" fsize={fsize} pos=({x},{y}) th={th} cx={cx}')

OUT = os.path.join(OUT_DIR, 'armed_text_only_v315q.jpg')
out_pil.save(OUT, quality=95)
print(f'[v315q] Saved {OUT}')

src_pil = Image.open(SRC).convert('RGB')
combined = Image.new('RGB', (src_pil.width * 2 + 30, src_pil.height), (50, 50, 50))
combined.paste(src_pil, (0, 0))
combined.paste(out_pil, (src_pil.width + 30, 0))
combined.save(os.path.join(OUT_DIR, 'armed_compare_v315q.jpg'), quality=90)
print('[v315q] Saved compare')
