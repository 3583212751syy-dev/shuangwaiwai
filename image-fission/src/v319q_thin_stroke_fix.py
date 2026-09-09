"""v319q: Fix thin-stroke text classification in text bands.

v319p bug: in text bands, CCs with h<60% band_h (e.g. thin strokes of
'I', 'E', 'T') were misclassified as 'chain' and preserved as black
residue. They are NOT chain links; they are text.

Distinguish by morphology:
  - Chain link: w/h ~ 1 (round), w <= 15, area < 200
  - Text thin stroke: high aspect ratio (w/h > 2.5) OR width > 15
                     OR area > 200

So: in text band, if h < 60% band_h AND (w/h > 2.5 OR w > 15 OR area > 200)
  -> treat as text (fill)
  -> else: treat as chain link (preserve)
"""
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt

PROJECT = Path("E:/Desktop/双接口/image-fission")
SRC = PROJECT / "jobs" / "v315" / "source_armed.jpg"
OUT_DIR = PROJECT / "jobs" / "v319q"
ARIAL_BLACK = r"C:/Windows/Fonts/ariblk.ttf"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_jpg(p):
    with open(p, 'rb') as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


print(f'[v319q] Loading source...')
src_bgr = load_jpg(str(SRC))
H, W = src_bgr.shape[:2]
img_rgb = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)

BANDS = [
    {'name': 'small', 'y0': 472, 'y1': 523, 'new_text': 'WE HONOR', 'orig_cx': 783, 'fsize': 71, 'spacing': 11, 'orig_w': 517},
    {'name': 'big1',  'y0': 541, 'y1': 640, 'new_text': 'BRAVE',    'orig_cx': 829, 'fsize': 138, 'spacing': 198, 'orig_w': 1322},
    {'name': 'big2',  'y0': 656, 'y1': 755, 'new_text': 'LEGION',   'orig_cx': 863, 'fsize': 138, 'spacing': 65, 'orig_w': 917},
]

# Step 1: Build masks with shape-aware classification
letter_mask = np.zeros((H, W), dtype=bool)
chain_mask = np.zeros((H, W), dtype=bool)

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
        x0 = stats[i, cv2.CC_STAT_LEFT]
        y_loc = stats[i, cv2.CC_STAT_TOP]
        aspect = w / max(h, 1)
        is_letter = (h >= band_h * 0.6) and (a >= 1000)
        # Thin stroke: high aspect OR wide OR big area (text artifact, not chain link)
        is_thin_stroke = (h < band_h * 0.6) and ((aspect > 2.5) or (w > 15) or (a > 200))
        if is_letter or is_thin_stroke:
            letter_mask[y0+y_loc:y0+y_loc+h, x0:x0+w] = True
        else:
            cc = (labels[y_loc:y_loc+h, x0:x0+w] == i)
            chain_mask[y0+y_loc:y0+y_loc+h, x0:x0+w] |= cc

print(f'[v319q] letter_mask px: {int(letter_mask.sum())}, chain_mask px: {int(chain_mask.sum())}')

# Step 2: distance_transform_edt fill
source_pixels = ~letter_mask & ~chain_mask
_, nearest_idx = distance_transform_edt(~source_pixels, return_distances=True, return_indices=True)
cleaned_rgb = img_rgb.copy()
cleaned_rgb[letter_mask] = img_rgb[nearest_idx[0][letter_mask], nearest_idx[1][letter_mask]]
print(f'[v319q] Distance transform fill done')

# Step 3: Anti-alias ring NS inpaint (kept from v319p, this step is light)
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

if ring_mask.any():
    bgr = cv2.cvtColor(cleaned_rgb, cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(bgr, (ring_mask.astype(np.uint8) * 255), 1, cv2.INPAINT_NS)
    cleaned_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
    print(f'[v319q] Ring inpaint done (px: {int(ring_mask.sum())})')

Image.fromarray(cleaned_rgb).save(str(OUT_DIR / 'armed_lama_cleaned.png'))

# Step 4: Draw text (one-pass per band, simple positioning)
out_pil = Image.fromarray(cleaned_rgb)
draw = ImageDraw.Draw(out_pil)

for band in BANDS:
    text = band['new_text']
    fsize = band['fsize']
    spacing = band['spacing']
    orig_cx = band['orig_cx']
    y1_band = band['y1']
    font = ImageFont.truetype(ARIAL_BLACK, fsize)
    ascent, descent = font.getmetrics()
    bbox = draw.textbbox((0, 0), text, font=font, spacing=spacing)
    total_w = bbox[2] - bbox[0]
    paste_x = orig_cx - total_w // 2 - bbox[0]
    # baseline = paste_y + ascent; cap bottom = baseline (no descender in caps)
    # so to put cap bottom at y1_band: paste_y = y1_band - ascent
    paste_y = y1_band - ascent
    draw.text((paste_x, paste_y), text, font=font, fill=(20, 20, 20), spacing=spacing)
    print(f'[v319q] {band["name"]}: "{text}" paste=({paste_x},{paste_y}) total_w={total_w} ascent={ascent}')

OUT = OUT_DIR / 'armed_text_only_v319q.jpg'
out_pil.save(str(OUT), quality=95)

# Compare
src_pil = Image.open(str(SRC)).convert('RGB')
combined = Image.new('RGB', (src_pil.width * 2 + 30, src_pil.height), (50, 50, 50))
combined.paste(src_pil, (0, 0))
combined.paste(out_pil.convert('RGB'), (src_pil.width + 30, 0))
combined.save(str(OUT_DIR / 'armed_compare_v319q.jpg'), quality=90)

src_crop = src_pil.crop((0, 440, W, 780))
res_crop = out_pil.crop((0, 440, W, 780))
band_compare = Image.new('RGB', (src_crop.width * 2 + 20, src_crop.height), (60, 60, 60))
band_compare.paste(src_crop, (0, 0))
band_compare.paste(res_crop, (src_crop.width + 20, 0))
band_compare.save(str(OUT_DIR / 'armed_text_band_compare_v319q.jpg'), quality=95)
print(f'[v319q] Done')
