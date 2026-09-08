"""v319o: Fix Y positioning bug + loosen small-band threshold.

Bug 1: Small-band letter CCs (E/S/P/T) have full band height but the CC
       detection misses them because the small band is only 51px tall and
       h>=60%*51=30.6px threshold + area>=1000 misses thin strokes. Result:
       ghost letters "E S PP T TH" remain.

Bug 2: Y positioning uses bbox[3]-bbox[1] (cap height) which is correct
       for the formula, but the formula itself was OK. Need to verify.

Fix:
  - Loosen small-band threshold: h>=40 (not 30.6) + a>=300 (not 1000)
  - For big bands, keep strict h>=60%*99=59.4 + a>=1000
  - Add anti-alias ring with NS inpaint radius=1 (same as v319n)
  - Verify Y alignment by checking actual text pixel y-range
"""
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt

PROJECT = Path("E:/Desktop/双接口/image-fission")
SRC = PROJECT / "jobs" / "v315" / "source_armed.jpg"
OUT_DIR = PROJECT / "jobs" / "v319o"
ARIAL_BLACK = r"C:/Windows/Fonts/ariblk.ttf"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_jpg(p):
    with open(p, 'rb') as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


print(f'[v319o] Loading source...')
src_bgr = load_jpg(str(SRC))
H, W = src_bgr.shape[:2]
print(f'  source: {W}x{H}')
img_rgb = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)

# Per-band letter thresholds (loosened for small band)
BANDS = [
    {'name': 'small', 'y0': 472, 'y1': 523, 'min_h': 40, 'min_a': 300, 'new_text': 'WE HONOR', 'orig_cx': 783, 'fsize': 71, 'spacing': 11, 'orig_w': 517},
    {'name': 'big1',  'y0': 541, 'y1': 640, 'min_h': 59, 'min_a': 1000, 'new_text': 'BRAVE',    'orig_cx': 829, 'fsize': 138, 'spacing': 198, 'orig_w': 1322},
    {'name': 'big2',  'y0': 656, 'y1': 755, 'min_h': 59, 'min_a': 1000, 'new_text': 'LEGION',   'orig_cx': 863, 'fsize': 138, 'spacing': 65, 'orig_w': 917},
]

# Step 1: Build letter_mask and chain_mask
letter_mask = np.zeros((H, W), dtype=bool)
chain_mask = np.zeros((H, W), dtype=bool)

for band in BANDS:
    y0, y1 = band['y0'], band['y1']
    band_h = y1 - y0
    band_gray = gray[y0:y1, :]
    dark = (band_gray < 80).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    for i in range(1, n):
        h = stats[i, cv2.CC_STAT_HEIGHT]
        a = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        x0 = stats[i, cv2.CC_STAT_LEFT]
        y_loc = stats[i, cv2.CC_STAT_TOP]
        if h >= band['min_h'] and a >= band['min_a']:
            letter_mask[y0+y_loc:y0+y_loc+h, x0:x0+w] = True
        else:
            cc = (labels[y_loc:y_loc+h, x0:x0+w] == i)
            chain_mask[y0+y_loc:y0+y_loc+h, x0:x0+w] |= cc

print(f'[v319o] letter_mask px: {int(letter_mask.sum())}')
print(f'[v319o] chain_mask px: {int(chain_mask.sum())}')

# Per-band letter counts
for band in BANDS:
    y0, y1 = band['y0'], band['y1']
    print(f'  {band["name"]} y={y0}-{y1}: letter_px={int(letter_mask[y0:y1,:].sum())} chain_px={int(chain_mask[y0:y1,:].sum())}')

# Step 2: distance_transform_edt fill
source_pixels = ~letter_mask & ~chain_mask
print(f'[v319o] source_pixels px: {int(source_pixels.sum())}')

_, nearest_idx = distance_transform_edt(~source_pixels, return_distances=True, return_indices=True)
src_y = nearest_idx[0]
src_x = nearest_idx[1]

cleaned_rgb = img_rgb.copy()
cleaned_rgb[letter_mask] = img_rgb[src_y[letter_mask], src_x[letter_mask]]
print(f'[v319o] Distance transform fill done')

# Step 3: Anti-alias ring inpaint (NS radius=1)
ring_mask = np.zeros((H, W), dtype=bool)
for band in BANDS:
    y0, y1 = band['y0'], band['y1']
    band_letter = letter_mask[y0:y1, :]
    if not band_letter.any():
        continue
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    band_dilated = cv2.dilate(band_letter.astype(np.uint8), k, iterations=1).astype(bool)
    band_ring = band_dilated & ~band_letter & ~chain_mask[y0:y1, :]
    band_gray = gray[y0:y1, :]
    band_ring &= (band_gray >= 80) & (band_gray <= 160)
    ring_mask[y0:y1, :] |= band_ring

print(f'[v319o] anti-alias ring px: {int(ring_mask.sum())}')

if ring_mask.any():
    ring_uint8 = (ring_mask.astype(np.uint8) * 255)
    bgr = cv2.cvtColor(cleaned_rgb, cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(bgr, ring_uint8, 1, cv2.INPAINT_NS)
    cleaned_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
    print(f'[v319o] Anti-alias ring inpaint done')

Image.fromarray(cleaned_rgb).save(str(OUT_DIR / 'armed_lama_cleaned.png'))
print(f'[v319o] Saved cleaned base')

# Step 4: Write new text with VERIFIED Y positioning
out_pil = Image.fromarray(cleaned_rgb)
draw = ImageDraw.Draw(out_pil)

for band in BANDS:
    text = band['new_text']
    fsize = band['fsize']
    spacing = band['spacing']
    orig_cx = band['orig_cx']
    y0_band = band['y0']
    y1_band = band['y1']
    font = ImageFont.truetype(ARIAL_BLACK, fsize)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]  # cap height
    cap_top = bbox[1]
    cap_bottom = bbox[3]
    n_chars = len(text)
    total_w = text_w + spacing * (n_chars - 1) if spacing > 0 else text_w
    paste_x = orig_cx - total_w // 2 - bbox[0]
    # Bottom-align: cap_bottom at y1_band
    paste_y = y1_band - cap_bottom
    print(f'[v319o] {band["name"]}: bbox={bbox} cap_top={cap_top} cap_bottom={cap_bottom} cap_h={text_h}')
    print(f'  paste=({paste_x},{paste_y}) total_w={total_w} expected_y_range=[{paste_y+cap_top},{paste_y+cap_bottom}] band=[{y0_band},{y1_band}]')

    txt_layer = Image.new('RGBA', (total_w + 20, cap_bottom - cap_top + 20), (0, 0, 0, 0))
    txt_draw = ImageDraw.Draw(txt_layer)
    cur_x = 10 - bbox[0]
    for ch in text:
        txt_draw.text((cur_x, 10 - cap_top), ch, font=font, fill=(20, 20, 20, 255))
        ch_bbox = font.getbbox(ch)
        cur_x += (ch_bbox[2] - ch_bbox[0]) + spacing
    out_pil.paste(txt_layer, (paste_x, paste_y), txt_layer)

OUT = OUT_DIR / 'armed_text_only_v319o.jpg'
out_pil.save(str(OUT), quality=95)
print(f'[v319o] Saved {OUT}')

# Verify text Y position by finding dark pixels in output near each band
for band in BANDS:
    y0, y1 = band['y0'] - 30, band['y1'] + 30
    out_band = np.array(out_pil.crop((0, y0, W, y1)).convert('RGB'))
    out_gray = cv2.cvtColor(out_band, cv2.COLOR_BGR2GRAY)
    dark = (out_gray < 80).sum(axis=1)
    in_text = dark > 30
    if in_text.any():
        first = np.where(in_text)[0][0] + y0
        last = np.where(in_text)[0][-1] + y0
        print(f'[v319o] {band["name"]}: text y={first}..{last} band y={band["y0"]}..{band["y1"]}')

src_pil = Image.open(str(SRC)).convert('RGB')
combined = Image.new('RGB', (src_pil.width * 2 + 30, src_pil.height), (50, 50, 50))
combined.paste(src_pil, (0, 0))
combined.paste(out_pil.convert('RGB'), (src_pil.width + 30, 0))
combined.save(str(OUT_DIR / 'armed_compare_v319o.jpg'), quality=90)

src_crop = src_pil.crop((0, 440, W, 780))
res_crop = out_pil.crop((0, 440, W, 780))
band_compare = Image.new('RGB', (src_crop.width * 2 + 20, src_crop.height), (60, 60, 60))
band_compare.paste(src_crop, (0, 0))
band_compare.paste(res_crop, (src_crop.width + 20, 0))
band_compare.save(str(OUT_DIR / 'armed_text_band_compare_v319o.jpg'), quality=95)
print(f'[v319o] Saved compare images')
