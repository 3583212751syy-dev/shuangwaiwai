"""
v315n: text-only fission, robust cleanup.

Bug in v315k: camo sample was taken from rows 30px above/below the band,
which lands INSIDE other text bands (e.g. big1 samples y=511..541 = "WE SUPPORT THE").
Result: scaled original text bleeds through every band.

Fix in v315n:
  1. Mask = all anti-aliased text pixels (gray<200) dilated 15px - chain CCs
  2. ZERO OUT those regions in the source (so cv2.inpaint can't copy them)
  3. cv2.inpaint NS radius=12 on the whole image (full camo context)
  4. Write new text with smaller font + tight centering on band midpoint
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
print(f'[v315n] Loaded {W}x{H}')

BANDS = [
    {'name': 'small', 'y0': 472, 'y1': 523, 'min_h': 25, 'min_w': 12, 'min_a': 200,  'dilate_px': 14, 'target_h': 38},
    {'name': 'big1',  'y0': 541, 'y1': 640, 'min_h': 60, 'min_w': 30, 'min_a': 1000, 'dilate_px': 18, 'target_h': 90},
    {'name': 'big2',  'y0': 656, 'y1': 755, 'min_h': 60, 'min_w': 30, 'min_a': 1000, 'dilate_px': 18, 'target_h': 90},
]

# Step 1: classify CCs into text vs chain per band
chain_mask = np.zeros((H, W), dtype=np.uint8)
for band in BANDS:
    y0, y1 = band['y0'], band['y1']
    band_gray = gray[y0:y1, :]
    dark = (band_gray < 75).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    for i in range(1, n):
        w, h, a = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        if not (w >= band['min_w'] and h >= band['min_h'] and a >= band['min_a']):
            # chain CC
            x0 = stats[i, cv2.CC_STAT_LEFT]
            y_loc = stats[i, cv2.CC_STAT_TOP]
            cc = (labels[y_loc:y_loc+h, x0:x0+w] == i)
            chain_mask[y0+y_loc:y0+y_loc+h, x0:x0+w][cc] = 255

# Step 2: build mask = anti-aliased text (gray<200) + dilate - chain
mask_full = np.zeros((H, W), dtype=np.uint8)
for band in BANDS:
    y0, y1 = band['y0'], band['y1']
    band_gray = gray[y0:y1, :]
    # Use gray<200 to catch anti-aliased edges of text
    band_text = (band_gray < 200).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                   (2 * band['dilate_px'] + 1, 2 * band['dilate_px'] + 1))
    band_dilated = cv2.dilate(band_text, k, iterations=1)
    # Subtract chain (never paint over chain)
    band_dilated[chain_mask[y0:y1, :] > 0] = 0
    mask_full[y0:y1, :] = np.maximum(mask_full[y0:y1, :], band_dilated)

# Step 3: zero out the masked region in source
img_bgr_zeroed = img_bgr.copy()
img_bgr_zeroed[mask_full > 0] = 0

# Step 4: inpaint the whole image with NS radius 12
inpainted = cv2.inpaint(img_bgr_zeroed, mask_full, 12, cv2.INPAINT_NS)
img_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
print(f'[v315n] Inpainted {int((mask_full > 0).sum())} pixels')

# Step 5: write new text — smaller, band-centered, no fake text fitting
out_pil = Image.fromarray(img_rgb)
draw = ImageDraw.Draw(out_pil)

# For each band: find font size that gives text height <= target_h
def get_font(target_h, font_path, sample='ABCDEFG'):
    for size in range(30, 200):
        font = ImageFont.truetype(font_path, size)
        bbox = draw.textbbox((0, 0), sample, font=font)
        h = bbox[3] - bbox[1]
        if h >= target_h:
            return font, size
    return ImageFont.truetype(font_path, 200), 200

NEW_TEXTS = {
    'small': 'HONOR THOSE WHO SERVED',
    'big1':  'BRAVE',
    'big2':  'LEGION',
}

# Measure original text x-extent for centering
def orig_x_extent(gray, y0, y1, min_w, min_h, min_a):
    bg = gray[y0:y1, :]
    dark = (bg < 75).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    xs = []
    for i in range(1, n):
        w, h, a = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        if w >= min_w and h >= min_h and a >= min_a:
            xs.append(stats[i, cv2.CC_STAT_LEFT])
            xs.append(stats[i, cv2.CC_STAT_LEFT] + w)
    return (min(xs), max(xs)) if xs else (0, 0)

for band in BANDS:
    text = NEW_TEXTS[band['name']]
    font, fsize = get_font(band['target_h'], FONT_PATH)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    ox0, ox1 = orig_x_extent(gray, band['y0'], band['y1'],
                             band['min_w'], band['min_h'], band['min_a'])
    cx = (ox0 + ox1) // 2
    cy = (band['y0'] + band['y1']) // 2
    x = cx - tw // 2 - bbox[0]
    y = cy - th // 2 - bbox[1]
    draw.text((x, y), text, font=font, fill=(20, 20, 20))
    print(f'[v315n] {band["name"]}: "{text}" fsize={fsize} pos=({x},{y}) th={th}')

OUT = os.path.join(OUT_DIR, 'armed_text_only_v315n.jpg')
out_pil.save(OUT, quality=95)
print(f'[v315n] Saved {OUT}')

src_pil = Image.open(SRC).convert('RGB')
combined = Image.new('RGB', (src_pil.width * 2 + 30, src_pil.height), (50, 50, 50))
combined.paste(src_pil, (0, 0))
combined.paste(out_pil, (src_pil.width + 30, 0))
combined.save(os.path.join(OUT_DIR, 'armed_compare_v315n.jpg'), quality=90)
print('[v315n] Saved compare')
