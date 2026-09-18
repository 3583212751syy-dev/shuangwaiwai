"""
v316d: final clean version.

Lessons from v316a/b/c:
  - Anton font at fsize where nat_h ≈ band_h works
  - Scale text horizontally to fill orig_w
  - Position: paste_y = band center - new_h//2
  - But also reduce font size slightly to leave 5-10px margin from band edges

New text choices (closer to original word counts):
  - small "WE SUPPORT THE" 16 chars → "HONOR OUR HEROES" 16 chars
  - big1  "ARMED" 5 chars → "BRAVE" 5 chars (same length!)
  - big2  "FORCES" 6 chars → "LEGION" 6 chars (same length!)

This gives similar letter count → similar width.

Also, position VETERANS/LEGION with extra padding from band edges:
  - new_h * 0.95 of band_h, leave 5% margin (5px top/bot)

Vertical: use y_text_top = band_y0 + small_pad to align top of text
with top of original text.
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
    {'name': 'small', 'y0': 472, 'y1': 523, 'orig_w': 517,  'new_text': 'HEROES'},
    {'name': 'big1',  'y0': 541, 'y1': 640, 'orig_w': 1322, 'new_text': 'BRAVE'},
    {'name': 'big2',  'y0': 656, 'y1': 755, 'orig_w': 917,  'new_text': 'LEGION'},
]

# Step 1: chain mask = non-full-height CCs
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
        is_text = (h >= band_h * 0.7) and (a >= 800)
        x0 = stats[i, cv2.CC_STAT_LEFT]
        y_loc = stats[i, cv2.CC_STAT_TOP]
        if is_text:
            text_mask_global[y0+y_loc:y0+y_loc+h, x0:x0+w] = True
        else:
            cc = (labels[y_loc:y_loc+h, x0:x0+w] == i)
            chain_mask[y0+y_loc:y0+y_loc+h, x0:x0+w][cc] = 255

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
print(f'[v316d] filled {n_text} text px, {int(use_median.sum())} median-capped')

# Step 3: Write new text — carefully positioned
out_pil = Image.fromarray(img_rgb)
draw = ImageDraw.Draw(out_pil)

# Draw a temporary surface to measure text bbox
tmp_surf = Image.new('RGB', (W, H))
tmp_draw = ImageDraw.Draw(tmp_surf)

for band in BANDS:
    text = band['new_text']
    target_h = band['y1'] - band['y0']
    # Find font size so nat_h ≈ target_h * 0.92 (leave 8% margin)
    fsize = 50
    for fs in range(50, 200):
        font = ImageFont.truetype(ANTON_PATH, fs)
        bbox = tmp_draw.textbbox((0, 0), text, font=font, anchor='lt')
        nat_h = bbox[3] - bbox[1]
        if nat_h >= target_h * 0.92:
            fsize = fs
            break
    bbox = tmp_draw.textbbox((0, 0), text, font=font, anchor='lt')
    nat_w = bbox[2] - bbox[0]
    nat_h = bbox[3] - bbox[1]
    # Scale x to fill orig_w
    sx = band['orig_w'] / nat_w
    # Clamp sx to avoid extreme stretching (1.0 = no stretch, max 2.5)
    sx = max(0.5, min(sx, 2.5))
    new_w = int(nat_w * sx)
    new_h = int(nat_h)

    # Render text on transparent layer at original size
    txt_layer = Image.new('RGBA', (nat_w + 20, nat_h + 20), (0, 0, 0, 0))
    txt_draw = ImageDraw.Draw(txt_layer)
    txt_draw.text((-bbox[0] + 10, -bbox[1] + 10), text, font=font, fill=(20, 20, 20, 255))
    # Resize horizontally
    txt_layer_scaled = txt_layer.resize((new_w, new_h), Image.LANCZOS)

    # Position: centered to original text center x (W/2)
    paste_x = (W - new_w) // 2
    # Position: centered to band center y
    paste_y = (band['y0'] + band['y1']) // 2 - new_h // 2

    out_pil.paste(txt_layer_scaled, (paste_x, paste_y), txt_layer_scaled)
    print(f'[v316d] {band["name"]}: "{text}" fsize={fsize} sx={sx:.2f} paste=({paste_x},{paste_y}) size={new_w}x{new_h}')

OUT = os.path.join(OUT_DIR, 'armed_text_only_v316d.jpg')
out_pil.convert('RGB').save(OUT, quality=95)

src_pil = Image.open(SRC).convert('RGB')
combined = Image.new('RGB', (src_pil.width * 2 + 30, src_pil.height), (50, 50, 50))
combined.paste(src_pil, (0, 0))
combined.paste(out_pil.convert('RGB'), (src_pil.width + 30, 0))
combined.save(os.path.join(OUT_DIR, 'armed_compare_v316d.jpg'), quality=90)
print(f'[v316d] Saved {OUT}')
