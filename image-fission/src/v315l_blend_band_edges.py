"""
v315l_blend_band_edges.py
v315k with edge feathering so the camo patch blends seamlessly with
surrounding camo. Uses alpha blending at the top and bottom of each band.
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
print(f'[v315l] Loaded {W}x{H}')

BANDS = [
    {'name': 'small', 'y0': 472, 'y1': 523, 'min_h_chain': 25, 'min_w_chain': 12, 'min_area_chain': 200, 'x_pad': 30, 'dilate_px': 12},
    {'name': 'big1',  'y0': 541, 'y1': 640, 'min_h_chain': 60, 'min_w_chain': 30, 'min_area_chain': 1000, 'x_pad': 30, 'dilate_px': 14},
    {'name': 'big2',  'y0': 656, 'y1': 755, 'min_h_chain': 60, 'min_w_chain': 30, 'min_area_chain': 1000, 'x_pad': 30, 'dilate_px': 14},
]

# Build masks
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
    band_chain = chain_mask[y0:y1, pad:W-pad]
    dilated[band_chain > 0] = 0
    fill_mask[y0:y1, pad:W-pad] = np.maximum(fill_mask[y0:y1, pad:W-pad], dilated)

print(f'[v315l] fill_mask pixels: {int((fill_mask>0).sum())}')

# For each band: use a wider sample region that includes context above and below
# Then do alpha blending at band edges with surrounding camo
EDGE_FADE_PX = 20

for band in BANDS:
    y0, y1 = band['y0'], band['y1']
    band_h = y1 - y0
    band_fill = fill_mask[y0:y1, :]

    if band_fill.sum() == 0:
        continue

    # Build camo sample: take a vertical strip that includes the band + some above + some below
    # Then crop to band height — this gives natural lighting match
    sample_top = max(0, y0 - 60)
    sample_bot = min(H, y1 + 60)
    camo_with_context = img_rgb[sample_top:sample_bot, :, :].copy()
    # Replace text area in the context with camo from a wider area (simple median fill)
    # For simplicity, just take the camo from the wider context
    # The middle of the context corresponds to the band
    sample_band_start = y0 - sample_top
    sample_band_end = y1 - sample_top
    # Take a "clean camo" from a wider range and tile/blend into band
    # Strategy: use the camo from the source image as-is, but at the band position,
    # camo was originally overwritten by text. We can copy camo from above or below the band.

    # Best approach: copy camo from immediately above the band (within context)
    # That's the camo at y0-60..y0-30 in the source (which is clean camo since text was below)
    # OR: use camo from y0+30..y1-30 if available, or from other text-free areas

    # For simplicity: take camo from 200..400 (well above all bands) and resize
    clean_camo = img_rgb[200:400, :, :]
    camo_resized = cv2.resize(clean_camo, (W, band_h), interpolation=cv2.INTER_LINEAR)

    # Compute alpha blend map for this band:
    # - Inside fill_mask: use camo_resized
    # - Outside fill_mask (within band): use original
    # - At band edges (top/bottom): blend with surrounding camo for seamless transition

    band_rgb = img_rgb[y0:y1, :, :].copy()
    band_orig = img_rgb[y0:y1, :, :].copy()

    # First: fill mask area with camo
    band_rgb[band_fill > 0] = camo_resized[band_fill > 0]

    # Edge feathering at band top/bottom:
    # At y=0 (top of band), alpha=0.5 (mix with original)
    # At y=EDGE_FADE_PX, alpha=1.0 (full camo)
    # ...similar for bottom

    alpha_map = np.ones((band_h, W, 3), dtype=np.float32)
    for i in range(EDGE_FADE_PX):
        f = i / EDGE_FADE_PX
        if i < band_h:
            alpha_map[i, :, :] = f
        if band_h - 1 - i >= 0:
            alpha_map[band_h - 1 - i, :, :] = f

    # Only blend at edges where fill_mask is 0 (outside text area)
    for i in range(EDGE_FADE_PX):
        if i < band_h:
            row_top = alpha_map[i, :, :]
            # Where fill_mask is 0 (camo already), blend with original
            no_fill_top = (band_fill[i, :] == 0)
            band_rgb[i, :, :] = (camo_resized[i, :, :] * row_top + band_orig[i, :, :] * (1 - row_top)).astype(np.uint8)
            # Where fill_mask is 255 (already camo from fill), no change needed

        if band_h - 1 - i >= 0:
            row_bot = alpha_map[band_h - 1 - i, :, :]
            no_fill_bot = (band_fill[band_h - 1 - i, :] == 0)
            band_rgb[band_h - 1 - i, :, :] = (camo_resized[band_h - 1 - i, :, :] * row_bot + band_orig[band_h - 1 - i, :, :] * (1 - row_bot)).astype(np.uint8)

    img_rgb[y0:y1, :, :] = band_rgb
    print(f'[v315l] Band {band["name"]}: filled + edge blend done')

# Write text
out_pil = Image.fromarray(img_rgb)
draw = ImageDraw.Draw(out_pil)

def get_font_for_height(target_h, font_path, sample_text='ABCDEFG'):
    for size in range(50, 200):
        font = ImageFont.truetype(font_path, size)
        bbox = draw.textbbox((0, 0), sample_text, font=font)
        actual_h = bbox[3] - bbox[1]
        if actual_h >= target_h:
            return font, size
    return ImageFont.truetype(font_path, 200), 200

FONT_SIZES = {}
for band in BANDS:
    band_h = band['y1'] - band['y0']
    font, size = get_font_for_height(band_h, FONT_PATH)
    FONT_SIZES[band['name']] = (font, size)

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
    font, fsize = FONT_SIZES[band['name']]
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
    print(f'[v315l] Band {band["name"]}: text="{text}" font={fsize}px pos=({x},{y}) orig_cx={orig_cx}')

OUT = os.path.join(OUT_DIR, 'armed_text_only_v315l.jpg')
out_pil.save(OUT, quality=95)
print(f'[v315l] Saved {OUT}')

src_pil = Image.open(SRC).convert('RGB')
combined = Image.new('RGB', (src_pil.width * 2 + 30, src_pil.height), (50, 50, 50))
combined.paste(src_pil, (0, 0))
combined.paste(out_pil, (src_pil.width + 30, 0))
combined.save(os.path.join(OUT_DIR, 'armed_compare_v315l.jpg'), quality=90)
print('[v315l] Saved compare')