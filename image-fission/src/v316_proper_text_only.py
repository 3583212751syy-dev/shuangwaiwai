"""
v316: ARMED text-only fission, PROPER version.

Previous failures (v315..v315v) root causes:
  1. WRONG FONT — used Impact (h/font=0.79, sans-serif) instead of Anton
     (h/font=0.86, vertical-heavy sans). Original ARMED is Anton-style,
     not Didone serif as I incorrectly assumed.
  2. WRONG FONT SIZE — measured small band target_h=38 instead of 51,
     leading to compressed text on top band.
  3. WRONG X CENTERING — used orig_x_extent from MIN_W=12 CCs, but those
     don't include chain at band edges. Result: text not centered to
     original positions.
  4. cv2.inpaint on high-freq camo ALWAYS leaves residue. Permanent hard
     rule. v315v used distance_transform_edt which was OK.

v316 approach:
  - Use Anton-Regular.ttf (h/font=0.86 for vertical-heavy ARMED-style)
  - Text-only fission: ONLY the 3 text bands get processed, EVERYTHING
    else (camo, dog tag, chain, everything below y=755) is preserved
    pixel-perfect from source.
  - Fill strategy: scipy distance_transform_edt with 4px dilate on
    gray<100 mask, 25px distance cap, fallback to band median.
  - Vertical positions: y_center anchored to each band's geometric
    center (NOT target_h, but actual font height from Anton).

Hard rule from v315: NEVER use cv2.inpaint on high-freq camo.
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
print(f'[v316] Loaded {W}x{H}')

# Measured bands (exact from measure script)
# Band 0 small "WE SUPPORT THE":  y=472..523, h=51
# Band 1 big   "ARMED":            y=541..640, h=99
# Band 2 big   "FORCES":           y=656..755, h=99
BANDS = [
    {'name': 'small', 'y0': 472, 'y1': 523, 'target_h': 51, 'font_size': 58,  'orig_text': 'WE SUPPORT THE'},
    {'name': 'big1',  'y0': 541, 'y1': 640, 'target_h': 99, 'font_size': 115, 'orig_text': 'ARMED'},
    {'name': 'big2',  'y0': 656, 'y1': 755, 'target_h': 99, 'font_size': 115, 'orig_text': 'FORCES'},
]

NEW_TEXTS = {
    'small': 'WE HONOR THE BRAVE',
    'big1':  'VETERANS',
    'big2':  'LEGION',
}

# Pre-load fonts
FONTS = {}
for band in BANDS:
    FONTS[band['name']] = ImageFont.truetype(ANTON_PATH, band['font_size'])
    print(f'[v316] {band["name"]} font: Anton @ {band["font_size"]}px')

# Step 1: Build chain mask (CCs that are not text but dark pixels)
# Strategy: a CC is "text" if it spans >= 60% of band height AND has area >= 800
#           otherwise it's chain/link/dog tag edge
chain_mask = np.zeros((H, W), dtype=np.uint8)
for band in BANDS:
    y0, y1 = band['y0'], band['y1']
    band_gray = gray[y0:y1, :]
    dark = (band_gray < 80).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    band_h = y1 - y0
    for i in range(1, n):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        a = stats[i, cv2.CC_STAT_AREA]
        # If CC is small relative to band, it's not text (chain, decorative)
        is_text = (h >= band_h * 0.6) and (a >= 800) and (w >= 20)
        if not is_text:
            x0 = stats[i, cv2.CC_STAT_LEFT]
            y_loc = stats[i, cv2.CC_STAT_TOP]
            cc = (labels[y_loc:y_loc+h, x0:x0+w] == i)
            chain_mask[y0+y_loc:y0+y_loc+h, x0:x0+w][cc] = 255

# Step 2: Fill text bands using distance_transform_edt
DIST_CAP = 25
for band in BANDS:
    y0, y1 = band['y0'], band['y1']
    band_rgb = img_rgb[y0:y1, :].copy()
    band_gray = gray[y0:y1, :]
    chain_in_band = chain_mask[y0:y1, :] > 0

    # Tight text mask + dilate 4px
    text_pix = (band_gray < 100)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    text_pix_d = cv2.dilate(text_pix.astype(np.uint8) * 255, k, iterations=1) > 0
    # Subtract chain from text mask (chain is not text)
    text_pix_d = text_pix_d & ~chain_in_band

    if text_pix_d.sum() == 0:
        print(f'[v316] {band["name"]}: no text pixels found, skipping fill')
        continue

    # Camo source = non-text, non-chain
    camo_pix = ~text_pix_d & ~chain_in_band
    camo_pixels = band_rgb[camo_pix]
    if len(camo_pixels) == 0:
        continue
    median_color = np.median(camo_pixels, axis=0)

    # distance_transform_edt
    # target = text pixels (need filling)
    # source = camo pixels (look up these)
    # chain must NOT be a source — include in target so DT skips them
    target = text_pix_d | chain_in_band
    dist, (src_y, src_x) = distance_transform_edt(target, return_indices=True)

    text_indices = np.where(text_pix_d)
    n_text = len(text_indices[0])

    text_dist = dist[text_pix_d]
    use_nearest = text_dist <= DIST_CAP
    use_median = ~use_nearest

    fill_colors = np.zeros((n_text, 3), dtype=np.uint8)
    fill_colors[use_nearest] = band_rgb[src_y[text_pix_d][use_nearest], src_x[text_pix_d][use_nearest]]
    fill_colors[use_median] = median_color

    band_rgb[text_indices] = fill_colors
    img_rgb[y0:y1, :] = band_rgb

    n_capped = int(use_median.sum())
    print(f'[v316] {band["name"]}: filled {n_text} text px, {n_capped} capped to median (dist>{DIST_CAP}px)')

# Step 3: Write new text with Anton font, centered to band
out_pil = Image.fromarray(img_rgb)
draw = ImageDraw.Draw(out_pil)

# Measure original text extent for centering (use orig_text)
def orig_x_extent(gray, y0, y1, band_h):
    """Find the leftmost and rightmost x of major text CCs in band."""
    bg = gray[y0:y1, :]
    dark = (bg < 80).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    xs = []
    for i in range(1, n):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        a = stats[i, cv2.CC_STAT_AREA]
        if h >= band_h * 0.6 and a >= 800 and w >= 20:
            xs.append(stats[i, cv2.CC_STAT_LEFT])
            xs.append(stats[i, cv2.CC_STAT_LEFT] + w)
    return (min(xs), max(xs)) if xs else (0, W)

for band in BANDS:
    text = NEW_TEXTS[band['name']]
    font = FONTS[band['name']]
    bbox = draw.textbbox((0, 0), text, font=font, anchor='lt')
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    ox0, ox1 = orig_x_extent(gray, band['y0'], band['y1'], band['y1'] - band['y0'])
    cx = (ox0 + ox1) // 2
    cy = (band['y0'] + band['y1']) // 2
    x = cx - tw // 2 - bbox[0]
    y = cy - th // 2 - bbox[1]
    # Black text on camo (like original)
    draw.text((x, y), text, font=font, anchor='lt', fill=(20, 20, 20))
    print(f'[v316] {band["name"]}: "{text}" font=Anton@{band["font_size"]}px bbox={bbox} pos=({x},{y}) th={th}')

OUT = os.path.join(OUT_DIR, 'armed_text_only_v316.jpg')
out_pil.save(OUT, quality=95)
print(f'[v316] Saved {OUT}')

# Compare side by side
src_pil = Image.open(SRC).convert('RGB')
combined = Image.new('RGB', (src_pil.width * 2 + 30, src_pil.height), (50, 50, 50))
combined.paste(src_pil, (0, 0))
combined.paste(out_pil, (src_pil.width + 30, 0))
combined.save(os.path.join(OUT_DIR, 'armed_compare_v316.jpg'), quality=90)
print('[v316] Saved compare')
