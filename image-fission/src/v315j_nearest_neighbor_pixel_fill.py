"""
v315j_nearest_neighbor_pixel_fill.py
For each text pixel, find the nearest non-text (camo) pixel in the SAME BAND
and copy its color. This is the cleanest possible inpaint - preserves camo
texture exactly, just replaces text pixels with similar nearby camo pixels.

Strategy:
  1. Build "fill" mask (text pixels in bands) and "preserve" mask (chain pixels)
  2. For each fill pixel, find nearest preserve-or-source pixel using
     cv2.distanceTransformWithLabels with DIST_LABEL_PIXEL
  3. Map label → coordinate of source pixel
  4. Copy color from source pixel to fill pixel
  5. Write new text
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
print(f'[v315j] Loaded {W}x{H}')

BANDS = [
    {'name': 'small', 'y0': 472, 'y1': 523, 'min_h_chain': 25, 'min_w_chain': 12, 'min_area_chain': 200, 'x_pad': 30, 'dilate_px': 6},
    {'name': 'big1',  'y0': 541, 'y1': 640, 'min_h_chain': 60, 'min_w_chain': 30, 'min_area_chain': 1000, 'x_pad': 30, 'dilate_px': 12},
    {'name': 'big2',  'y0': 656, 'y1': 755, 'min_h_chain': 60, 'min_w_chain': 30, 'min_area_chain': 1000, 'x_pad': 30, 'dilate_px': 12},
]

# Build text mask (text pixels) and chain mask
text_only_mask = np.zeros((H, W), dtype=np.uint8)
chain_mask = np.zeros((H, W), dtype=np.uint8)

for band in BANDS:
    y0, y1 = band['y0'], band['y1']
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
        cc_mask = (labels[y_local:y_local+h, x_local:x_local+w] == i)
        full_x0 = pad + x_local
        full_y0 = y0 + y_local
        if is_text:
            text_only_mask[full_y0:full_y0+h, full_x0:full_x0+w][cc_mask] = 255
        else:
            chain_mask[full_y0:full_y0+h, full_x0:full_x0+w][cc_mask] = 255

# Dilate text mask to capture anti-aliased edges
fill_mask = np.zeros((H, W), dtype=np.uint8)
for band in BANDS:
    y0, y1 = band['y0'], band['y1']
    pad = band['x_pad']
    dilate_px = band['dilate_px']
    band_text = text_only_mask[y0:y1, pad:W-pad]
    if band_text.sum() == 0:
        continue
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilate_px+1, 2*dilate_px+1))
    dilated = cv2.dilate(band_text, k, iterations=1)
    # Don't include chain pixels
    band_chain = chain_mask[y0:y1, pad:W-pad]
    dilated[band_chain > 0] = 0
    fill_mask[y0:y1, pad:W-pad] = np.maximum(fill_mask[y0:y1, pad:W-pad], dilated)

print(f'[v315j] fill_mask pixels: {int((fill_mask>0).sum())}')

# Per-pixel nearest-neighbor fill
# source_pixels = NOT fill AND NOT chain
# For each fill pixel, find nearest source pixel and copy color

source_pixels = ((fill_mask == 0) & (chain_mask == 0)).astype(np.uint8)
print(f'[v315j] source_pixels: {int(source_pixels.sum())}')

# Process each band separately to keep labels manageable
result_rgb = img_rgb.copy()
for band in BANDS:
    y0, y1 = band['y0'], band['y1']
    band_fill = fill_mask[y0:y1, :]
    band_chain = chain_mask[y0:y1, :]
    band_source = ((band_fill == 0) & (band_chain == 0)).astype(np.uint8)

    if band_fill.sum() == 0:
        continue

    band_h = y1 - y0

    # Distance transform with labels
    # For each zero pixel in band_source (= fill pixel), gives label of nearest source pixel
    dist, nearest_labels = cv2.distanceTransformWithLabels(
        band_source, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )

    # Get unique labels (corresponds to source pixel indices in some scan order)
    # Actually DIST_LABEL_PIXEL labels each source pixel with a unique number
    # Need to find which (y, x) each label corresponds to

    # Strategy: for each unique label, find one source pixel position
    # Then build label → color map

    # Use the labels directly: for each fill pixel, nearest_labels[y, x] gives label of nearest source
    # We need: for each label value, find the (y, x) of ONE source pixel with that label

    # Build label → source position map by scanning band_source
    source_positions = {}
    for sy in range(band_h):
        for sx in range(W):
            if band_source[sy, sx]:
                label = nearest_labels[sy, sx]
                if label not in source_positions:
                    source_positions[label] = (sy, sx)

    # Now for each fill pixel, look up source color
    band_rgb = img_rgb[y0:y1, :, :].copy()
    for fy in range(band_h):
        for fx in range(W):
            if band_fill[fy, fx]:
                label = nearest_labels[fy, fx]
                if label in source_positions:
                    sy, sx = source_positions[label]
                    band_rgb[fy, fx] = band_rgb[sy, sx]

    result_rgb[y0:y1, :, :] = band_rgb
    print(f'[v315j] Band {band["name"]}: per-pixel NN fill done ({int(band_fill.sum())} pixels)')

img_rgb = result_rgb

# Write new text
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
    print(f'[v315j] Band {band["name"]}: text="{text}" font={fsize}px '
          f'pos=({x},{y}) orig_cx={orig_cx}')

OUT = os.path.join(OUT_DIR, 'armed_text_only_v315j.jpg')
out_pil.save(OUT, quality=95)
print(f'[v315j] Saved {OUT}')

src_pil = Image.open(SRC).convert('RGB')
combined = Image.new('RGB', (src_pil.width * 2 + 30, src_pil.height), (50, 50, 50))
combined.paste(src_pil, (0, 0))
combined.paste(out_pil, (src_pil.width + 30, 0))
combined.save(os.path.join(OUT_DIR, 'armed_compare_v315j.jpg'), quality=90)
print('[v315j] Saved compare')