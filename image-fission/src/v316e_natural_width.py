"""
v316e: clean approach. No text stretching. Anton font natural width.

Lessons from v316a-d:
  - Don't stretch Anton horizontally (sx>1.5 makes letter spacing awful)
  - Don't shrink Anton horizontally (sx<0.5 makes it squashed)
  - Just use Anton at natural aspect ratio
  - Position: center x to band center, center y to band center
  - Use EXACT band heights (51/99/99) to size the font
  - Leave 2px vertical margin from band edges

This is the cleanest approach. The text won't be as wide as the original
ARMED (which used a different narrower font), but it will be CLEAR,
LEGIBLE, and POSITIONALLY CORRECT.
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

BANDS = [
    {'name': 'small', 'y0': 472, 'y1': 523, 'new_text': 'HEROES'},
    {'name': 'big1',  'y0': 541, 'y1': 640, 'new_text': 'BRAVE'},
    {'name': 'big2',  'y0': 656, 'y1': 755, 'new_text': 'LEGION'},
]

# Step 1: text vs chain classification
# Text CC = letter vertical body (h >= 50% band_h, area >= 800)
#        + letter horizontal serif (w >= 30, h >= 20, area >= 1500, h < 50% band_h)
# Chain CC = everything else (chain links, decorative dots)
chain_mask = np.zeros((H, W), dtype=np.uint8)
text_mask_global = np.zeros((H, W), dtype=bool)
for band in BANDS:
    y0, y1 = band['y0'], band['y1']
    band_h = y1 - y0
    band_gray = gray[y0:y1, :]
    dark = (band_gray < 80).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    for i in range(1, n):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        a = stats[i, cv2.CC_STAT_AREA]
        # Vertical letter body: spans most of band height (ignore area — thin
        # strokes of small-band letters can have tiny area but full height)
        is_letter = (h >= band_h * 0.35)
        # Horizontal serif (sits within letter, like A's crossbar):
        # wide, short, large area, NOT a full-height letter
        is_serif = (w >= 30) and (h >= 20) and (a >= 1500) and (not is_letter)
        is_text = is_letter or is_serif
        x0 = stats[i, cv2.CC_STAT_LEFT]
        y_loc = stats[i, cv2.CC_STAT_TOP]
        if is_text:
            text_mask_global[y0+y_loc:y0+y_loc+h, x0:x0+w] = True
        else:
            cc = (labels[y_loc:y_loc+h, x0:x0+w] == i)
            chain_mask[y0+y_loc:y0+y_loc+h, x0:x0+w][cc] = 255
    print(f'[v316e] {band["name"]} band: classified text vs chain CCs')

# Step 2: dilate + scipy DT
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
text_mask_d = cv2.dilate(text_mask_global.astype(np.uint8) * 255, k, iterations=1) > 0
text_mask_d = text_mask_d & (chain_mask == 0)

DIST_CAP = 15
target = text_mask_d | (chain_mask > 0)
dist, (src_y, src_x) = distance_transform_edt(target, return_indices=True)
camo_mask = ~text_mask_d & (chain_mask == 0)
camo_pixels = img_rgb[camo_mask]
median_color = np.median(camo_pixels, axis=0)
text_indices = np.where(text_mask_d)
n_text = len(text_indices[0])
text_dist = dist[text_mask_d]
use_nearest = text_dist <= DIST_CAP
use_median = ~use_nearest
fill_colors = np.zeros((n_text, 3), dtype=np.uint8)
fill_colors[use_nearest] = img_rgb[src_y[text_mask_d][use_nearest], src_x[text_mask_d][use_nearest]]
fill_colors[use_median] = median_color
img_rgb[text_indices] = fill_colors
print(f'[v316e] filled {n_text} text px, {int(use_median.sum())} median-capped')

# Step 3: Write new text — natural width, centered
out_pil = Image.fromarray(img_rgb)
draw = ImageDraw.Draw(out_pil)

tmp_surf = Image.new('RGB', (W, H))
tmp_draw = ImageDraw.Draw(tmp_surf)

for band in BANDS:
    text = band['new_text']
    target_h = band['y1'] - band['y0']
    # Find font size so nat_h matches target_h
    fsize = 50
    for fs in range(50, 250):
        font = ImageFont.truetype(ANTON_PATH, fs)
        bbox = tmp_draw.textbbox((0, 0), text, font=font, anchor='lt')
        nat_h = bbox[3] - bbox[1]
        if nat_h >= target_h:
            fsize = fs
            break
    bbox = tmp_draw.textbbox((0, 0), text, font=font, anchor='lt')
    nat_w = bbox[2] - bbox[0]
    nat_h = bbox[3] - bbox[1]
    # Position centered at band center
    paste_x = (W - nat_w) // 2 - bbox[0]
    paste_y = (band['y0'] + band['y1']) // 2 - nat_h // 2 - bbox[1]
    draw.text((paste_x, paste_y), text, font=font, anchor='lt', fill=(20, 20, 20))
    print(f'[v316e] {band["name"]}: "{text}" fsize={fsize} nat={nat_w}x{nat_h} paste=({paste_x},{paste_y})')

OUT = os.path.join(OUT_DIR, 'armed_text_only_v316e.jpg')
out_pil.save(OUT, quality=95)

src_pil = Image.open(SRC).convert('RGB')
combined = Image.new('RGB', (src_pil.width * 2 + 30, src_pil.height), (50, 50, 50))
combined.paste(src_pil, (0, 0))
combined.paste(out_pil, (src_pil.width + 30, 0))
combined.save(os.path.join(OUT_DIR, 'armed_compare_v316e.jpg'), quality=90)

# Crop text band region for inspection
src_crop = src_pil.crop((0, 440, 1556, 780))
res_crop = out_pil.crop((0, 440, 1556, 780))
band_compare = Image.new('RGB', (src_crop.width * 2 + 20, src_crop.height), (60, 60, 60))
band_compare.paste(src_crop, (0, 0))
band_compare.paste(res_crop, (src_crop.width + 20, 0))
band_compare.save(os.path.join(OUT_DIR, 'armed_text_band_compare_v316e.jpg'), quality=95)
print(f'[v316e] Saved {OUT}')
