"""v319p: Simplify text rendering - draw whole string at once.

v319o bug: per-char drawing caused text bbox to extend beyond RGBA layer bounds,
making actual text height 99px instead of expected 53px (cap height).

Fix: draw the whole text string with PIL.draw.text() at once, position by
the baseline (ascent). For bottom-align cap to band y1:
  paste_y = band_y1 - font.getmetrics()[0]   (ascent)
Then cap is at [paste_y + bbox[1], paste_y + bbox[3]] in image coordinates.

Also: keep v319o's loosened small-band mask (h>=40, a>=300).
"""
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt

PROJECT = Path("E:/Desktop/双接口/image-fission")
SRC = PROJECT / "jobs" / "v315" / "source_armed.jpg"
OUT_DIR = PROJECT / "jobs" / "v319p"
ARIAL_BLACK = r"C:/Windows/Fonts/ariblk.ttf"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_jpg(p):
    with open(p, 'rb') as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


print(f'[v319p] Loading source...')
src_bgr = load_jpg(str(SRC))
H, W = src_bgr.shape[:2]
img_rgb = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)

BANDS = [
    {'name': 'small', 'y0': 472, 'y1': 523, 'min_h': 40, 'min_a': 300, 'new_text': 'WE HONOR', 'orig_cx': 783, 'fsize': 71, 'spacing': 11, 'orig_w': 517},
    {'name': 'big1',  'y0': 541, 'y1': 640, 'min_h': 59, 'min_a': 1000, 'new_text': 'BRAVE',    'orig_cx': 829, 'fsize': 138, 'spacing': 198, 'orig_w': 1322},
    {'name': 'big2',  'y0': 656, 'y1': 755, 'min_h': 59, 'min_a': 1000, 'new_text': 'LEGION',   'orig_cx': 863, 'fsize': 138, 'spacing': 65, 'orig_w': 917},
]

# Step 1: Build masks
letter_mask = np.zeros((H, W), dtype=bool)
chain_mask = np.zeros((H, W), dtype=bool)

for band in BANDS:
    y0, y1 = band['y0'], band['y1']
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

print(f'[v319p] letter_mask px: {int(letter_mask.sum())}, chain_mask px: {int(chain_mask.sum())}')

# Step 2: distance_transform_edt fill
source_pixels = ~letter_mask & ~chain_mask
_, nearest_idx = distance_transform_edt(~source_pixels, return_distances=True, return_indices=True)
cleaned_rgb = img_rgb.copy()
cleaned_rgb[letter_mask] = img_rgb[nearest_idx[0][letter_mask], nearest_idx[1][letter_mask]]
print(f'[v319p] Distance transform fill done')

# Step 3: Anti-alias ring NS inpaint
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
    print(f'[v319p] Ring inpaint done (px: {int(ring_mask.sum())})')

Image.fromarray(cleaned_rgb).save(str(OUT_DIR / 'armed_lama_cleaned.png'))

# Step 4: Draw text with SIMPLE per-band positioning
out_pil = Image.fromarray(cleaned_rgb)
draw = ImageDraw.Draw(out_pil)

for band in BANDS:
    text = band['new_text']
    fsize = band['fsize']
    spacing = band['spacing']
    orig_cx = band['orig_cx']
    y1_band = band['y1']
    font = ImageFont.truetype(ARIAL_BLACK, fsize)
    ascent = font.getmetrics()[0]

    # Draw whole text in one call with letter spacing via anchor
    # PIL's draw.text supports 'spacing' parameter for letter spacing
    # Calculate total width manually
    bbox = draw.textbbox((0, 0), text, font=font, spacing=spacing)
    total_w = bbox[2] - bbox[0]
    paste_x = orig_cx - total_w // 2 - bbox[0]
    paste_y = y1_band - ascent  # cap-bottom at band y1

    draw.text((paste_x, paste_y), text, font=font, fill=(20, 20, 20), spacing=spacing)
    print(f'[v319p] {band["name"]}: "{text}" paste=({paste_x},{paste_y}) total_w={total_w} cap_range=[{paste_y + font.getbbox(text)[1]},{paste_y + font.getbbox(text)[3]}] band_y1={y1_band}')

OUT = OUT_DIR / 'armed_text_only_v319p.jpg'
out_pil.save(str(OUT), quality=95)

# Verify Y position
for band in BANDS:
    y_check0, y_check1 = band['y0'] - 30, band['y1'] + 30
    out_band = np.array(out_pil.crop((0, y_check0, W, y_check1)).convert('RGB'))
    out_gray = cv2.cvtColor(out_band, cv2.COLOR_BGR2GRAY)
    dark = (out_gray < 80).sum(axis=1)
    in_text = dark > 30
    if in_text.any():
        first = np.where(in_text)[0][0] + y_check0
        last = np.where(in_text)[0][-1] + y_check0
        print(f'[v319p] VERIFY {band["name"]}: text y={first}..{last} band y={band["y0"]}..{band["y1"]}')

src_pil = Image.open(str(SRC)).convert('RGB')
combined = Image.new('RGB', (src_pil.width * 2 + 30, src_pil.height), (50, 50, 50))
combined.paste(src_pil, (0, 0))
combined.paste(out_pil.convert('RGB'), (src_pil.width + 30, 0))
combined.save(str(OUT_DIR / 'armed_compare_v319p.jpg'), quality=90)

src_crop = src_pil.crop((0, 440, W, 780))
res_crop = out_pil.crop((0, 440, W, 780))
band_compare = Image.new('RGB', (src_crop.width * 2 + 20, src_crop.height), (60, 60, 60))
band_compare.paste(src_crop, (0, 0))
band_compare.paste(res_crop, (src_crop.width + 20, 0))
band_compare.save(str(OUT_DIR / 'armed_text_band_compare_v319p.jpg'), quality=95)
print(f'[v319p] Done')
