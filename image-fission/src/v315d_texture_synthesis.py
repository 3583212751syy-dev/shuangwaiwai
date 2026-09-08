"""
v315d_texture_synthesis.py
Use clean camo sample to replace text band areas instead of cv2.inpaint.
This avoids cv2.inpaint's NS/TELEA artifacts (which produce dark ghost shapes
when text edges bleed into the fill region).

Approach:
  1. For each text band, sample clean camo from above the band
  2. Resize camo sample to match band height
  3. Composite camo into the band area, EXCLUDING chain pixels (chain preserved)
  4. Light edge feathering so the seam isn't visible
  5. Write new text at original positions with Impact font
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
print(f'[v315d] Loaded {W}x{H}')

# ---- Band positions ----
BANDS = [
    {'name': 'small', 'y0': 472, 'y1': 523, 'min_h_chain': 25, 'min_w_chain': 12, 'min_area_chain': 200, 'x_pad': 30},
    {'name': 'big1',  'y0': 541, 'y1': 640, 'min_h_chain': 60, 'min_w_chain': 30, 'min_area_chain': 1000, 'x_pad': 30},
    {'name': 'big2',  'y0': 656, 'y1': 755, 'min_h_chain': 60, 'min_w_chain': 30, 'min_area_chain': 1000, 'x_pad': 30},
]

# ---- Identify chain pixels in each band ----
def find_chain_mask(gray, band, H, W):
    """Find chain CCs in the band — small dark components not matching text char geometry."""
    y0, y1 = band['y0'], band['y1']
    pad = band['x_pad']
    band_gray = gray[y0:y1, pad:W-pad]
    dark_pix = (band_gray < 75).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark_pix, connectivity=8)
    chain_mask = np.zeros((H, W), dtype=np.uint8)
    for i in range(1, n):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        a = stats[i, cv2.CC_STAT_AREA]
        x_local = stats[i, cv2.CC_STAT_LEFT]
        y_local = stats[i, cv2.CC_STAT_TOP]
        is_chain = not (w >= band['min_w_chain'] and h >= band['min_h_chain'] and a >= band['min_area_chain'])
        if is_chain:
            chain_mask[y0 + y_local : y0 + y_local + h, pad + x_local : pad + x_local + w][labels[y_local:y_local+h, x_local:x_local+w] == i] = 255
    # Dilate slightly so chain looks continuous (the chain links have small gaps)
    chain_mask = cv2.dilate(chain_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    return chain_mask

# ---- For each band: sample clean camo, composite into band area ----
# Strategy: take camo from y=100..460 (above small band) — clean, same vertical section
CAMO_SAMPLE_Y0 = 100
CAMO_SAMPLE_Y1 = 460  # 360px of clean camo

# Also sample a bit from below dog tag for better color match in lower band
CAMO_BELOW_Y0 = 900
CAMO_BELOW_Y1 = 1200

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

for band in BANDS:
    y0, y1 = band['y0'], band['y1']
    band_h = y1 - y0

    # Pick camo source based on band position
    if band['name'] in ('small', 'big1'):
        camo_src = img_rgb[CAMO_SAMPLE_Y0:CAMO_SAMPLE_Y1, :, :]
    else:  # big2
        # Use camo from below the dog tag — closer in vertical position to FORCES
        camo_src = img_rgb[CAMO_BELOW_Y0:CAMO_BELOW_Y1, :, :]

    # Resize camo source to match band dimensions
    camo_resized = cv2.resize(camo_src, (W, band_h), interpolation=cv2.INTER_LINEAR)

    # Find chain pixels in this band
    chain_mask = find_chain_mask(gray, band, H, W)
    band_chain_mask = chain_mask[y0:y1, :]

    # Build composite: where chain_mask > 0, keep original; else replace with camo
    original_band = img_rgb[y0:y1, :, :].copy()
    composite = camo_resized.copy()
    composite[band_chain_mask > 0] = original_band[band_chain_mask > 0]

    # Write back
    img_rgb[y0:y1, :, :] = composite
    print(f'[v315d] Band {band["name"]}: replaced (h={band_h}), chain_pixels={int((band_chain_mask>0).sum())}')

# ---- Write new text ----
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
    print(f'[v315d] Band {band["name"]}: text="{text}" font={fsize}px '
          f'pos=({x},{y}) orig_cx={orig_cx}')

OUT = os.path.join(OUT_DIR, 'armed_text_only_v315d.jpg')
out_pil.save(OUT, quality=95)
print(f'[v315d] Saved {OUT}')

# Compare
src_pil = Image.open(SRC).convert('RGB')
combined = Image.new('RGB', (src_pil.width * 2 + 30, src_pil.height), (50, 50, 50))
combined.paste(src_pil, (0, 0))
combined.paste(out_pil, (src_pil.width + 30, 0))
combined.save(os.path.join(OUT_DIR, 'armed_compare_v315d.jpg'), quality=90)
print('[v315d] Saved compare')