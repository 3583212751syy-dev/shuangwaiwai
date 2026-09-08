"""
v315h_final_cleanup.py
Builds on v315g but adds a final cleanup pass: any remaining dark spots in
the result (introduced by inpaint) get a small-radius inpaint to clean up.
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
print(f'[v315h] Loaded {W}x{H}')

BANDS = [
    {'name': 'small', 'y0': 472, 'y1': 523, 'min_h_chain': 25, 'min_w_chain': 12, 'min_area_chain': 200, 'x_pad': 30, 'dilate_px': 6},
    {'name': 'big1',  'y0': 541, 'y1': 640, 'min_h_chain': 60, 'min_w_chain': 30, 'min_area_chain': 1000, 'x_pad': 30, 'dilate_px': 12},
    {'name': 'big2',  'y0': 656, 'y1': 755, 'min_h_chain': 60, 'min_w_chain': 30, 'min_area_chain': 1000, 'x_pad': 30, 'dilate_px': 12},
]

# Step A: build text mask (text + dilated edges, chain excluded)
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

# Dilate text mask per band
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

# Pass 1: inpaint the text area
img_bgr_now = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
inpainted = cv2.inpaint(img_bgr_now, fill_mask, 4, cv2.INPAINT_TELEA)
img_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
print('[v315h] Pass 1: text inpaint done')

# Pass 2: find any NEW dark spots in result that weren't in source (within band area)
# These are TELEA artifacts from the dark text edges bleeding in
res_gray = cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY)
for band in BANDS:
    y0, y1 = band['y0'], band['y1']
    pad = band['x_pad']
    src_band = gray[y0:y1, pad:W-pad]
    res_band = res_gray[y0:y1, pad:W-pad]

    # New dark = dark in result but not in source (within band, excluding chain)
    new_dark = ((res_band < 90) & (src_band >= 90)).astype(np.uint8) * 255
    new_dark[chain_mask[y0:y1, pad:W-pad] > 0] = 0  # don't touch chain

    # Dilate slightly to capture artifact edges
    new_dark_dilated = cv2.dilate(new_dark, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

    # Tiny inpaint radius to clean up
    img_bgr_now = cv2.cvtColor(img_rgb[y0:y1, pad:W-pad], cv2.COLOR_RGB2BGR)
    cleaned = cv2.inpaint(img_bgr_now, new_dark_dilated, 2, cv2.INPAINT_TELEA)
    img_rgb[y0:y1, pad:W-pad] = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)
    print(f'[v315h] Pass 2 {band["name"]}: cleaned {int((new_dark>0).sum())} new dark spots')

# Pass 3: write new text
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
    print(f'[v315h] Band {band["name"]}: text="{text}" font={fsize}px '
          f'pos=({x},{y}) orig_cx={orig_cx}')

OUT = os.path.join(OUT_DIR, 'armed_text_only_v315h.jpg')
out_pil.save(OUT, quality=95)
print(f'[v315h] Saved {OUT}')

src_pil = Image.open(SRC).convert('RGB')
combined = Image.new('RGB', (src_pil.width * 2 + 30, src_pil.height), (50, 50, 50))
combined.paste(src_pil, (0, 0))
combined.paste(out_pil, (src_pil.width + 30, 0))
combined.save(os.path.join(OUT_DIR, 'armed_compare_v315h.jpg'), quality=90)
print('[v315h] Saved compare')