"""
v315e_blend_edges.py
Replace text band content with sampled camo, then use cv2.inpaint ONLY on
band edges (top + bottom strips) to blend with surrounding camo.

This solves the visible horizontal seam line issue from v315d.

Strategy:
  1. For each band, sample clean camo from elsewhere (above bands for top 2, below for last)
  2. Composite sampled camo into band area, EXCLUDING chain pixels
  3. cv2.inpaint ONLY the band edge strips (top 8px + bottom 8px) to blend
     with surrounding camo
  4. Write new text at original positions
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
print(f'[v315e] Loaded {W}x{H}')

BANDS = [
    {'name': 'small', 'y0': 472, 'y1': 523, 'min_h_chain': 25, 'min_w_chain': 12, 'min_area_chain': 200, 'x_pad': 30,
     'camo_src': (100, 460)},
    {'name': 'big1',  'y0': 541, 'y1': 640, 'min_h_chain': 60, 'min_w_chain': 30, 'min_area_chain': 1000, 'x_pad': 30,
     'camo_src': (100, 460)},
    {'name': 'big2',  'y0': 656, 'y1': 755, 'min_h_chain': 60, 'min_w_chain': 30, 'min_area_chain': 1000, 'x_pad': 30,
     'camo_src': (900, 1200)},
]

EDGE_FADE_PX = 12  # inpaint zone at band top/bottom

def find_chain_mask_in_band(gray, band, H, W):
    """Find chain CCs in band (small dark components not matching text char geometry)."""
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

# Step 1: For each band, composite sampled camo into band area
for band in BANDS:
    y0, y1 = band['y0'], band['y1']
    band_h = y1 - y0
    cs0, cs1 = band['camo_src']
    camo_src = img_rgb[cs0:cs1, :, :]
    camo_resized = cv2.resize(camo_src, (W, band_h), interpolation=cv2.INTER_LINEAR)

    chain_mask = find_chain_mask_in_band(gray, band, H, W)
    original_band = img_rgb[y0:y1, :, :].copy()
    composite = camo_resized.copy()
    composite[chain_mask > 0] = original_band[chain_mask > 0]
    img_rgb[y0:y1, :, :] = composite
    print(f'[v315e] Band {band["name"]}: composite done, chain_px={int((chain_mask>0).sum())}')

# Step 2: cv2.inpaint on band edge strips (top + bottom) of each band
# Build mask = top EDGE_FADE_PX rows + bottom EDGE_FADE_PX rows for each band
edge_mask = np.zeros((H, W), dtype=np.uint8)
for band in BANDS:
    y0, y1 = band['y0'], band['y1']
    # Top strip
    edge_mask[max(0, y0 - EDGE_FADE_PX):y0, :] = 255
    # Bottom strip
    edge_mask[y1:min(H, y1 + EDGE_FADE_PX), :] = 255

# Apply chain preservation: don't inpaint chain pixels
chain_global_mask = np.zeros((H, W), dtype=np.uint8)
for band in BANDS:
    y0, y1 = band['y0'], band['y1']
    cm = find_chain_mask_in_band(gray, band, H, W)
    chain_global_mask[y0:y1, :] = np.maximum(chain_global_mask[y0:y1, :], cm)

edge_mask[chain_global_mask > 0] = 0  # don't inpaint chain pixels
print(f'[v315e] Edge inpaint mask pixels: {int((edge_mask>0).sum())}')

# Convert to BGR for cv2.inpaint
img_bgr_now = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)  # rgb -> bgr for cv2
# Actually img_rgb is RGB; cv2.inpaint expects BGR or grayscale. Use the BGR version.
img_bgr_for_inpaint = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
inpainted = cv2.inpaint(img_bgr_for_inpaint, edge_mask, 4, cv2.INPAINT_NS)
img_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
print('[v315e] Edge inpaint done')

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
    print(f'[v315e] Band {band["name"]}: text="{text}" font={fsize}px '
          f'pos=({x},{y}) orig_cx={orig_cx}')

OUT = os.path.join(OUT_DIR, 'armed_text_only_v315e.jpg')
out_pil.save(OUT, quality=95)
print(f'[v315e] Saved {OUT}')

# Compare
src_pil = Image.open(SRC).convert('RGB')
combined = Image.new('RGB', (src_pil.width * 2 + 30, src_pil.height), (50, 50, 50))
combined.paste(src_pil, (0, 0))
combined.paste(out_pil, (src_pil.width + 30, 0))
combined.save(os.path.join(OUT_DIR, 'armed_compare_v315e.jpg'), quality=90)
print('[v315e] Saved compare')