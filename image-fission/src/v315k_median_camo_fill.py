"""
v315k_median_camo_fill.py
Fill text area with the MEDIAN color of camo in that band (preserves lighting),
then write new text on top. The new text covers the filled area.

Strategy:
  1. For each band, compute median camo color from non-text, non-chain pixels
  2. Fill text area (text + dilated edges, minus chain) with that median
  3. Add slight noise/texture variation by mixing in nearby camo pixels
  4. Write new text on top — text height matches band height exactly
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
print(f'[v315k] Loaded {W}x{H}')

BANDS = [
    {'name': 'small', 'y0': 472, 'y1': 523, 'min_h_chain': 25, 'min_w_chain': 12, 'min_area_chain': 200, 'x_pad': 30, 'dilate_px': 12},
    {'name': 'big1',  'y0': 541, 'y1': 640, 'min_h_chain': 60, 'min_w_chain': 30, 'min_area_chain': 1000, 'x_pad': 30, 'dilate_px': 14},
    {'name': 'big2',  'y0': 656, 'y1': 755, 'min_h_chain': 60, 'min_w_chain': 30, 'min_area_chain': 1000, 'x_pad': 30, 'dilate_px': 14},
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

# Build fill mask (text + dilated edges, excluding chain)
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

print(f'[v315k] fill_mask pixels: {int((fill_mask>0).sum())}')

# For each band, compute median camo color and fill the mask with it
# Plus add some variance by mixing in nearby pixels
for band in BANDS:
    y0, y1 = band['y0'], band['y1']
    band_fill = fill_mask[y0:y1, :]
    if band_fill.sum() == 0:
        continue

    band_rgb = img_rgb[y0:y1, :, :]
    band_gray = gray[y0:y1, :]

    # Median color of non-fill, non-chain pixels in this band
    not_fill = (band_fill == 0) & (chain_mask[y0:y1, :] == 0)
    pixels = band_rgb[not_fill]  # (N, 3) array
    if len(pixels) == 0:
        continue
    median_color = np.median(pixels, axis=0).astype(np.uint8)
    # Stddev for variation
    stddev = np.std(pixels, axis=0).astype(np.float32)

    # Fill the band with median + local variation
    # Use per-pixel local mean from a wider sample for natural variation
    # Sample camo from immediately above and below the band
    sample_h = 30
    sample_top = img_rgb[max(0, y0-sample_h):y0, :, :] if y0 >= sample_h else None
    sample_bot = img_rgb[y1:min(H, y1+sample_h), :, :] if y1 + sample_h <= H else None
    samples = []
    if sample_top is not None:
        samples.append(sample_top)
    if sample_bot is not None:
        samples.append(sample_bot)
    if samples:
        # Use sample to get local mean
        sample_combined = np.concatenate(samples, axis=0)
        # Resize to band dimensions
        camo_resized = cv2.resize(sample_combined, (W, y1 - y0), interpolation=cv2.INTER_LINEAR)
    else:
        camo_resized = np.full((y1 - y0, W, 3), median_color, dtype=np.uint8)

    # Fill pixels with sampled camo
    band_filled = band_rgb.copy()
    band_filled[band_fill > 0] = camo_resized[band_fill > 0]
    img_rgb[y0:y1, :, :] = band_filled
    print(f'[v315k] Band {band["name"]}: filled {int(band_fill.sum())} px with sampled camo, median=({int(median_color[0])},{int(median_color[1])},{int(median_color[2])})')

# Write text — use font size that gives text height matching band height exactly
# Impact font: cap-height / font_size = 0.71, but bbox text height ratio = 0.81
# Band h=99 → want text h=99 → font_size = 99 / 0.81 = 122
# Band h=51 → font_size = 51 / 0.82 = 62
out_pil = Image.fromarray(img_rgb)
draw = ImageDraw.Draw(out_pil)

# First measure actual bbox for each text to get exact font sizing
def get_font_for_height(target_h, font_path, sample_text='ABCDEFG'):
    """Find font size that gives bbox height matching target_h."""
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
    print(f'[v315k] Band {band["name"]}: target_h={band_h}, font_size={size}, actual_h={draw.textbbox((0,0),"ABCDE",font=font)[3]-draw.textbbox((0,0),"ABCDE",font=font)[1]}')

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

    # Center text in band using bbox center
    x = orig_cx - tw // 2 - bbox[0]
    y_band_center = (band['y0'] + band['y1']) // 2
    y = y_band_center - th // 2 - bbox[1]

    draw.text((x, y), text, font=font, fill=(20, 20, 20))
    print(f'[v315k] Band {band["name"]}: text="{text}" font={fsize}px '
          f'pos=({x},{y}) orig_cx={orig_cx} text_h={th}')

OUT = os.path.join(OUT_DIR, 'armed_text_only_v315k.jpg')
out_pil.save(OUT, quality=95)
print(f'[v315k] Saved {OUT}')

src_pil = Image.open(SRC).convert('RGB')
combined = Image.new('RGB', (src_pil.width * 2 + 30, src_pil.height), (50, 50, 50))
combined.paste(src_pil, (0, 0))
combined.paste(out_pil, (src_pil.width + 30, 0))
combined.save(os.path.join(OUT_DIR, 'armed_compare_v315k.jpg'), quality=90)
print('[v315k] Saved compare')