"""
v315c_text_only_chain_preserve.py
- TELEA inpaint instead of NS (NS produces cloud artifacts on textured camo)
- Save via PIL (handles unicode paths)
- Test both small/large radii
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

def save_jpg(img_bgr, p, quality=95):
    """Save via PIL to avoid cv2.imwrite unicode issues."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(img_rgb).save(p, quality=quality)

SRC = r'E:/Desktop/双接口/image-fission/jobs/v315/source_armed.jpg'
OUT_DIR = r'E:/Desktop/双接口/image-fission/jobs/v315'
FONT_PATH = r'C:/Windows/Fonts/impact.ttf'

img_bgr = load_jpg(SRC)
H, W = img_bgr.shape[:2]
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
print(f'[v315c] Loaded {W}x{H}')

BANDS = [
    {'name': 'small', 'y0': 472, 'y1': 523, 'min_h_chain': 25, 'min_w_chain': 12, 'min_area_chain': 200, 'x_pad': 30},
    {'name': 'big1',  'y0': 541, 'y1': 640, 'min_h_chain': 60, 'min_w_chain': 30, 'min_area_chain': 1000, 'x_pad': 30},
    {'name': 'big2',  'y0': 656, 'y1': 755, 'min_h_chain': 60, 'min_w_chain': 30, 'min_area_chain': 1000, 'x_pad': 30},
]

# ---- Build full-band mask minus chain ----
mask = np.zeros((H, W), dtype=np.uint8)

for band in BANDS:
    y0, y1 = band['y0'], band['y1']
    pad = band['x_pad']

    band_mask = np.zeros((H, W), dtype=np.uint8)
    band_mask[y0:y1, pad:W-pad] = 255

    band_gray = gray[y0:y1, pad:W-pad]
    dark_pix = (band_gray < 75).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark_pix, connectivity=8)
    chain_subtract = np.zeros_like(band_mask)
    for i in range(1, n):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        a = stats[i, cv2.CC_STAT_AREA]
        x_local = stats[i, cv2.CC_STAT_LEFT]
        y_local = stats[i, cv2.CC_STAT_TOP]
        is_chain = not (w >= band['min_w_chain'] and h >= band['min_h_chain'] and a >= band['min_area_chain'])
        if is_chain:
            chain_subtract[y0 + y_local : y0 + y_local + h, pad + x_local : pad + x_local + w][labels[y_local:y_local+h, x_local:x_local+w] == i] = 255

    band_mask[chain_subtract > 0] = 0
    mask = np.maximum(mask, band_mask)
    print(f'[v315c] Band {band["name"]}: chain_subtract={int((chain_subtract>0).sum())}')

# Save mask via PIL
save_jpg(mask, os.path.join(OUT_DIR, 'mask_v315c.png'))

# ---- TELEA inpaint (smoother on textures than NS) ----
inp_telea_3 = cv2.inpaint(img_bgr, mask, 3, cv2.INPAINT_TELEA)
inp_telea_5 = cv2.inpaint(img_bgr, mask, 5, cv2.INPAINT_TELEA)
inp_telea_8 = cv2.inpaint(img_bgr, mask, 8, cv2.INPAINT_TELEA)
inp_ns_5 = cv2.inpaint(img_bgr, mask, 5, cv2.INPAINT_NS)

save_jpg(inp_telea_3, os.path.join(OUT_DIR, 'inp_TELEA_r3.png'))
save_jpg(inp_telea_5, os.path.join(OUT_DIR, 'inp_TELEA_r5.png'))
save_jpg(inp_telea_8, os.path.join(OUT_DIR, 'inp_TELEA_r8.png'))
save_jpg(inp_ns_5, os.path.join(OUT_DIR, 'inp_NS_r5.png'))
print('[v315c] Saved inpaint variants')

# ---- Compose final using TELEA radius 5 ----
inpainted_rgb = cv2.cvtColor(inp_telea_5, cv2.COLOR_BGR2RGB)
out_pil = Image.fromarray(inpainted_rgb)
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
    print(f'[v315c] Band {band["name"]}: text="{text}" font={fsize}px '
          f'pos=({x},{y}) orig_cx={orig_cx}')

OUT = os.path.join(OUT_DIR, 'armed_text_only_v315c.jpg')
out_pil.save(OUT, quality=95)
print(f'[v315c] Saved {OUT}')

# Compare
src_pil = Image.open(SRC).convert('RGB')
W2 = src_pil.width
H2 = src_pil.height
combined = Image.new('RGB', (W2 * 2 + 30, H2), (50, 50, 50))
combined.paste(src_pil, (0, 0))
combined.paste(out_pil, (W2 + 30, 0))
combined.save(os.path.join(OUT_DIR, 'armed_compare_v315c.jpg'), quality=90)
print(f'[v315c] Saved compare')