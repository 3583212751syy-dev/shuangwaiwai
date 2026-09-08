"""
v315i_seamless_clone.py
Use cv2.seamlessClone (Poisson gradient-domain blending) to seamlessly
integrate sampled camo into the band area. This eliminates seam lines.

Strategy:
  1. Sample clean camo from above/below bands
  2. Resize to band dimensions
  3. cv2.seamlessClone the camo into the band area
  4. Chain pixels preserved (excluded from clone mask)
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
print(f'[v315i] Loaded {W}x{H}')

BANDS = [
    {'name': 'small', 'y0': 472, 'y1': 523, 'min_h_chain': 25, 'min_w_chain': 12, 'min_area_chain': 200, 'x_pad': 30,
     'camo_src': (200, 460)},
    {'name': 'big1',  'y0': 541, 'y1': 640, 'min_h_chain': 60, 'min_w_chain': 30, 'min_area_chain': 1000, 'x_pad': 30,
     'camo_src': (200, 460)},
    {'name': 'big2',  'y0': 656, 'y1': 755, 'min_h_chain': 60, 'min_w_chain': 30, 'min_area_chain': 1000, 'x_pad': 30,
     'camo_src': (900, 1200)},
]

def find_chain_mask_in_band(gray, band, H, W):
    y0, y1 = band['y0'], band['y1']
    pad = band['x_pad']
    band_gray = gray[y0:y1, pad:W-pad]
    dark_pix = (band_gray < 75).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark_pix, connectivity=8)
    chain_mask = np.zeros((y1 - y0, W), dtype=np.uint8)
    for i in range(1, n):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        a = stats[i, cv2.CC_STAT_AREA]
        x_local = stats[i, cv2.CC_STAT_LEFT]
        y_local = stats[i, cv2.CC_STAT_TOP]
        is_chain = not (w >= band['min_w_chain'] and h >= band['min_h_chain'] and a >= band['min_area_chain'])
        if is_chain:
            chain_mask[y_local:y_local+h, pad + x_local:pad + x_local + w][labels[y_local:y_local+h, x_local:x_local+w] == i] = 255
    chain_mask = cv2.dilate(chain_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    return chain_mask

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

for band in BANDS:
    y0, y1 = band['y0'], band['y1']
    band_h = y1 - y0
    cs0, cs1 = band['camo_src']

    # Sample camo from source region
    camo_src = img_rgb[cs0:cs1, :, :]
    camo_resized = cv2.resize(camo_src, (W, band_h), interpolation=cv2.INTER_LINEAR)

    # Mask for seamless clone: include all of band except chain pixels
    chain_mask = find_chain_mask_in_band(gray, band, H, W)
    clone_mask = np.ones((band_h, W), dtype=np.uint8) * 255
    clone_mask[chain_mask > 0] = 0

    # Use cv2.seamlessClone to blend camo into band
    # Destination center is the center of the band
    center = (W // 2, y0 + band_h // 2)

    # seamlessClone expects uint8 BGR or RGB
    dst = img_rgb.copy()
    # Convert to BGR for cv2.seamlessClone
    camo_bgr = cv2.cvtColor(camo_resized, cv2.COLOR_RGB2BGR)
    dst_bgr = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    try:
        blended_bgr = cv2.seamlessClone(camo_bgr, dst_bgr, clone_mask, center, cv2.NORMAL_CLONE)
        img_rgb = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)
        print(f'[v315i] Band {band["name"]}: seamlessClone done')
    except cv2.error as e:
        print(f'[v315i] Band {band["name"]}: seamlessClone failed: {e}, using simple composite')
        # Fallback: simple composite
        composite = camo_resized.copy()
        composite[chain_mask > 0] = dst[y0:y1, :, :][chain_mask > 0]
        dst[y0:y1, :, :] = composite
        img_rgb = dst

# Save intermediate (before text)
out_pil = Image.fromarray(img_rgb)
out_pil.save(os.path.join(OUT_DIR, 'armed_no_text_v315i.jpg'), quality=95)
print('[v315i] Saved armed_no_text_v315i.jpg')

# Write new text
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
    print(f'[v315i] Band {band["name"]}: text="{text}" font={fsize}px '
          f'pos=({x},{y}) orig_cx={orig_cx}')

OUT = os.path.join(OUT_DIR, 'armed_text_only_v315i.jpg')
out_pil.save(OUT, quality=95)
print(f'[v315i] Saved {OUT}')

src_pil = Image.open(SRC).convert('RGB')
combined = Image.new('RGB', (src_pil.width * 2 + 30, src_pil.height), (50, 50, 50))
combined.paste(src_pil, (0, 0))
combined.paste(out_pil, (src_pil.width + 30, 0))
combined.save(os.path.join(OUT_DIR, 'armed_compare_v315i.jpg'), quality=90)
print('[v315i] Saved compare')