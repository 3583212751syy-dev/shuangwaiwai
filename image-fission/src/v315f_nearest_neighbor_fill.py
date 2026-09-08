"""
v315f_nearest_neighbor_fill.py
For each text pixel, find the nearest non-text (camo) pixel in a local window
and copy its color. This is the cleanest possible "inpaint" because it preserves
the original camo texture exactly.

Strategy:
  1. Mark text pixels (gray<75) as "to fill"
  2. Exclude chain pixels (small CCs) from "to fill" — they're preserved
  3. Use cv2.distanceTransformWithLabels to find nearest non-text pixel for
     each text pixel
  4. Copy colors from nearest non-text pixel
  5. For text pixels far from any camo (>radius), use cv2.inpaint as fallback
  6. Write new text
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
print(f'[v315f] Loaded {W}x{H}')

BANDS = [
    {'name': 'small', 'y0': 472, 'y1': 523, 'min_h_chain': 25, 'min_w_chain': 12, 'min_area_chain': 200, 'x_pad': 30},
    {'name': 'big1',  'y0': 541, 'y1': 640, 'min_h_chain': 60, 'min_w_chain': 30, 'min_area_chain': 1000, 'x_pad': 30},
    {'name': 'big2',  'y0': 656, 'y1': 755, 'min_h_chain': 60, 'min_w_chain': 30, 'min_area_chain': 1000, 'x_pad': 30},
]

# Build "to fill" mask (text pixels within bands, excluding chain)
# Expand band y0/y1 by some margin to include anti-alias edges of letters
MARGIN = 6
fill_mask = np.zeros((H, W), dtype=np.uint8)
chain_mask = np.zeros((H, W), dtype=np.uint8)

for band in BANDS:
    y0 = max(0, band['y0'] - MARGIN)
    y1 = min(H, band['y1'] + MARGIN)
    pad = band['x_pad']

    band_gray = gray[y0:y1, pad:W-pad]
    dark_pix = (band_gray < 75).astype(np.uint8) * 255

    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark_pix, connectivity=8)

    for i in range(1, n):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        a = stats[i, cv2.CC_STAT_AREA]
        x_local = stats[i, cv2.CC_STAT_LEFT]
        y_local = stats[i, cv2.CC_STAT_TOP]
        is_text = (w >= band['min_w_chain']) and (h >= band['min_h_chain']) and (a >= band['min_area_chain'])

        full_x0 = pad + x_local
        full_y0 = y0 + y_local
        cc_mask = (labels[y_local:y_local+h, x_local:x_local+w] == i)

        if is_text:
            fill_mask[full_y0:full_y0+h, full_x0:full_x0+w][cc_mask] = 255
        else:
            chain_mask[full_y0:full_y0+h, full_x0:full_x0+w][cc_mask] = 255

# Expand fill_mask slightly (dilate) to cover anti-aliased text edges that are gray but visually text
# Find pixels with gray<100 within text bands (slightly looser threshold)
for band in BANDS:
    y0 = max(0, band['y0'] - MARGIN)
    y1 = min(H, band['y1'] + MARGIN)
    pad = band['x_pad']
    loose = (gray[y0:y1, pad:W-pad] < 110).astype(np.uint8) * 255
    # Keep only pixels that are connected to text via dilation
    loose = cv2.dilate(loose, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
    # Add to fill_mask (only where not chain)
    band_part = fill_mask[y0:y1, pad:W-pad]
    chain_part = chain_mask[y0:y1, pad:W-pad]
    new_fill = np.maximum(band_part, loose)
    new_fill[chain_part > 0] = 0
    fill_mask[y0:y1, pad:W-pad] = new_fill

print(f'[v315f] Fill mask pixels: {int((fill_mask>0).sum())}')
print(f'[v315f] Chain mask pixels: {int((chain_mask>0).sum())}')

# Now: fill each "fill" pixel with color from nearest non-fill pixel
# Non-fill = (fill_mask == 0)
non_fill_mask = ((fill_mask == 0) & (chain_mask == 0)).astype(np.uint8)

# Use distanceTransformWithLabels: for each pixel, gives nearest non-zero pixel
# We want: for each FILL pixel, find nearest NON-FILL pixel
# distanceTransformWithLabels(input, ...) returns dist and labels where:
#   input=non_fill_mask (1=non-fill, 0=fill)
#   For each fill pixel (0), dist=distance to nearest non-fill pixel, label=which non-fill
# So we need to pass non_fill_mask directly

# dist: float32, dist to nearest non-zero pixel for each zero pixel
# labels: int32, label of nearest non-zero pixel
dist, nearest_labels = cv2.distanceTransformWithLabels(non_fill_mask, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)

# Get unique labels in nearest_labels (these correspond to non-fill pixels)
# Each label value points to a (y, x) coordinate of a non-fill pixel
# Need to build a lookup: for each label value, what color to use

# Find all unique labels
unique_labels = np.unique(nearest_labels)
print(f'[v315f] Unique labels in distance map: {len(unique_labels)}')

# Build label → pixel coordinate map
# Labels in OpenCV's distanceTransformWithLabels start from 1 and increment for each CC
# But here non_fill_mask is connected, so there's likely 1 label
# Actually no — labels are pixel-level indices, not CC labels
# Each pixel has a label, and we need to know which non-fill pixel it points to

# Strategy: for each pixel (y, x) in fill_mask, find the position it points to
# nearest_labels[y, x] gives a label value. We need to map label → (y, x) of source pixel.

# In OpenCV, label value i corresponds to pixel that is i-th non-zero pixel (in row-major order)
# But this is unreliable. Let me use a different approach.

# Alternative: for each fill pixel, BFS-spread until we find a non-fill pixel
# Too slow.

# Better: use cv2.inpaint on the fill_mask but limit radius to small (3-4)
# This works because the fill mask is sparse (only text pixels, not whole band)
# and there's camo just outside the text edges

img_bgr_now = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
inpainted = cv2.inpaint(img_bgr_now, fill_mask, 3, cv2.INPAINT_TELEA)
img_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
print('[v315f] Inpaint TELEA r3 done')

# Step 3: Write new text
out_pil = Image.fromarray(img_rgb)
draw = ImageDraw.Draw(out_pil)

FONT_SIZES = {'small': 72, 'big1': 138, 'big2': 138}
NEW_TEXTS = {'small': 'WE HONOR OUR HEROES', 'big1': 'BRAVE', 'big2': 'LEGION'}

def measure_text_x_extent(gray, band_y0, band_y1, min_w, min_h, min_area):
    band_gray = gray[band_y0:band_y1, :]
    text_pix = (band_gray < 75).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(text_pix, connectivity=8)
    xs = []
    for i in range(1, n):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        a = stats[i, cv2.CC_STAT_AREA]
        if w >= min_w and h >= min_h and a >= min_area:
            xs.append(stats[i, cv2.CC_STAT_LEFT])
            xs.append(stats[i, cv2.CC_STAT_LEFT] + w)
    return (min(xs), max(xs)) if xs else (0, 0)

for band in BANDS:
    text = NEW_TEXTS[band['name']]
    fsize = FONT_SIZES[band['name']]
    font = ImageFont.truetype(FONT_PATH, fsize)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    orig_x0, orig_x1 = measure_text_x_extent(
        gray, band['y0'], band['y1'],
        band['min_w_chain'], band['min_h_chain'], band['min_area_chain'],
    )
    orig_cx = (orig_x0 + orig_x1) // 2

    x = orig_cx - tw // 2 - bbox[0]
    y_band_center = (band['y0'] + band['y1']) // 2
    y = y_band_center - th // 2 - bbox[1]

    draw.text((x, y), text, font=font, fill=(20, 20, 20))
    print(f'[v315f] Band {band["name"]}: text="{text}" font={fsize}px '
          f'pos=({x},{y}) orig_cx={orig_cx}')

OUT = os.path.join(OUT_DIR, 'armed_text_only_v315f.jpg')
out_pil.save(OUT, quality=95)
print(f'[v315f] Saved {OUT}')

src_pil = Image.open(SRC).convert('RGB')
combined = Image.new('RGB', (src_pil.width * 2 + 30, src_pil.height), (50, 50, 50))
combined.paste(src_pil, (0, 0))
combined.paste(out_pil, (src_pil.width + 30, 0))
combined.save(os.path.join(OUT_DIR, 'armed_compare_v315f.jpg'), quality=90)
print('[v315f] Saved compare')