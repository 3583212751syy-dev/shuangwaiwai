"""
v316c: ARMED text-only fission, FINAL attempt.

v316/v316b issues:
  1. New text "VETERANS / LEGION" is too narrow — doesn't fill original
     text width (ARMED = 1322px wide, VETERANS at 115px = ~400px)
  2. Vertical overlap: VETERANS occupies y=540..640, LEGION occupies
     y=655..755 — they're ALMOST touching but I centered on band midpoint
     which puts VETERANS slightly above original ARMED position

v316c approach:
  1. Write new text, then SCALE horizontally so its width matches the
     original text width (pillow.LANCZOS resize on text-only layer)
  2. Use PADDING between bands (real ARMED has 16-18px gap between bands
     — the new VETERANS @ 115px is exactly that height so natural gap)
  3. Vertical position: anchor new text BASELINE to original baseline
     (i.e., text bottom = original bottom for big1/big2)
  4. Fill strategy: use FULL scipy distance_transform_edt across the
     entire image, not per-band, to avoid seams at band edges

Actually simpler approach to avoid seams:
  - Identify ALL text CCs across the image at once
  - Single big DT call on whole image
  - Source pixels = everything-not-text-not-chain

Let me try.
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
print(f'[v316c] Loaded {W}x{H}')

BANDS = [
    {'name': 'small', 'y0': 472, 'y1': 523, 'orig_w': 517,  'orig_text': 'WE SUPPORT THE', 'new_text': 'HONOR THE BRAVE'},
    {'name': 'big1',  'y0': 541, 'y1': 640, 'orig_w': 1322, 'orig_text': 'ARMED',          'new_text': 'VETERANS'},
    {'name': 'big2',  'y0': 656, 'y1': 755, 'orig_w': 917,  'orig_text': 'FORCES',         'new_text': 'LEGION'},
]

# Step 1: chain mask — full-height text CCs only
chain_mask = np.zeros((H, W), dtype=np.uint8)
text_mask_global = np.zeros((H, W), dtype=bool)
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
        is_text = (h >= band_h * 0.7) and (a >= 800)
        x0 = stats[i, cv2.CC_STAT_LEFT]
        y_loc = stats[i, cv2.CC_STAT_TOP]
        if is_text:
            text_mask_global[y0+y_loc:y0+y_loc+h, x0:x0+w] = True
        else:
            cc = (labels[y_loc:y_loc+h, x0:x0+w] == i)
            chain_mask[y0+y_loc:y0+y_loc+h, x0:x0+w][cc] = 255

# Step 2: dilate text mask
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
text_mask_d = cv2.dilate(text_mask_global.astype(np.uint8) * 255, k, iterations=1) > 0
text_mask_d = text_mask_d & (chain_mask == 0)

# Step 3: scipy distance_transform_edt on whole image
DIST_CAP = 15
target = text_mask_d | (chain_mask > 0)
dist, (src_y, src_x) = distance_transform_edt(target, return_indices=True)

# camo source: not text, not chain
camo_mask = ~text_mask_d & (chain_mask == 0)
camo_pixels = img_rgb[camo_mask]
median_color = np.median(camo_pixels, axis=0)

text_indices = np.where(text_mask_d)
n_text = len(text_indices[0])
text_dist = dist[text_mask_d]
use_nearest = text_dist <= DIST_CAP
use_median = ~use_nearest

fill_colors = np.zeros((n_text, 3), dtype=np.uint8)
fill_colors[use_nearest] = img_rgb[src_y[text_mask_d][use_nearest], src_x[text_mask_d][use_nearest]]
fill_colors[use_median] = median_color

img_rgb[text_indices] = fill_colors
print(f'[v316c] filled {n_text} text px, {int(use_median.sum())} capped to median (>{DIST_CAP}px)')

# Step 4: Write new text — scale to fill original width
out_pil = Image.fromarray(img_rgb)
draw = ImageDraw.Draw(out_pil)

for band in BANDS:
    text = band['new_text']
    # Find font size where natural height matches band height
    target_h = band['y1'] - band['y0']
    for fsize in range(80, 160):
        font = ImageFont.truetype(ANTON_PATH, fsize)
        bbox = draw.textbbox((0, 0), text, font=font, anchor='lt')
        nat_h = bbox[3] - bbox[1]
        if nat_h >= target_h * 0.95:
            break
    # Get natural width
    bbox = draw.textbbox((0, 0), text, font=font, anchor='lt')
    nat_w = bbox[2] - bbox[0]
    nat_h = bbox[3] - bbox[1]
    # Scale to fill original width: sx = orig_w / nat_w
    sx = band['orig_w'] / nat_w
    # Render text to separate layer, scale, paste
    # Compute scaled text image size
    new_w = int(nat_w * sx)
    new_h = int(nat_h)
    # Render text at original size on a transparent layer
    txt_layer = Image.new('RGBA', (nat_w + 20, nat_h + 20), (0, 0, 0, 0))
    txt_draw = ImageDraw.Draw(txt_layer)
    txt_draw.text((-bbox[0], -bbox[1]), text, font=font, fill=(20, 20, 20, 255))
    # Resize horizontally
    txt_layer_scaled = txt_layer.resize((new_w, new_h), Image.LANCZOS)
    # Paste centered at band center
    cx_band = (band['y0'] + band['y1']) // 2
    # x: centered to original text x-center (which is W/2 for ARMED since it spans full width)
    cx = W // 2
    paste_x = cx - new_w // 2
    paste_y = cx_band - new_h // 2
    out_pil.paste(txt_layer_scaled, (paste_x, paste_y), txt_layer_scaled)
    print(f'[v316c] {band["name"]}: "{text}" fsize={fsize} sx={sx:.2f} paste=({paste_x},{paste_y}) size={new_w}x{new_h}')

OUT = os.path.join(OUT_DIR, 'armed_text_only_v316c.jpg')
out_pil.convert('RGB').save(OUT, quality=95)

src_pil = Image.open(SRC).convert('RGB')
combined = Image.new('RGB', (src_pil.width * 2 + 30, src_pil.height), (50, 50, 50))
combined.paste(src_pil, (0, 0))
combined.paste(out_pil, (src_pil.width + 30, 0))
combined.save(os.path.join(OUT_DIR, 'armed_compare_v316c.jpg'), quality=90)
print(f'[v316c] Saved {OUT}')
