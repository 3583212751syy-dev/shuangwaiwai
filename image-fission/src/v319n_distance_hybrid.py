"""v319n: Final attempt. Drop LaMa (rebuilds whole ARMED image).

Pipeline (combines best of v315v distance transform + v316e mask classification):
  1. Build letter mask: CCs with h >= 60% band_h (strict)
  2. Build chain mask: dark CCs not in letter (preserved)
  3. Letter region: scipy distance_transform_edt nearest-camo fill (cap=8px)
     - takes the closest non-mask pixel (chain or clean camo) as fill source
  4. Anti-alias ring (dilate letter mask 1-2px, gray 80-160):
     - cv2.inpaint NS radius=1 (gentle, only thin ring, no mosaic)
  5. Write new text with Arial Black at original Y/X positions
  6. PIL paste (RGBA -> RGB on the cleaned base)

Why this is the LAST attempt:
  - LaMa rebuilds ARMED whole image (Fourier conv can't see "clean" reference)
  - SDXL inpaint also rebuilds (diffusion can't maintain camo color)
  - cv2.inpaint alone always leaves residue on text edges
  - Hybrid: distance transform for letter body + thin NS ring for AA edges
"""
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt

PROJECT = Path("E:/Desktop/双接口/image-fission")
SRC = PROJECT / "jobs" / "v315" / "source_armed.jpg"
OUT_DIR = PROJECT / "jobs" / "v319n"
ARIAL_BLACK = r"C:/Windows/Fonts/ariblk.ttf"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_jpg(p):
    with open(p, 'rb') as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


print(f'[v319n] Loading source...')
src_bgr = load_jpg(str(SRC))
H, W = src_bgr.shape[:2]
print(f'  source: {W}x{H}')
img_rgb = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)

BANDS = [
    {'name': 'small', 'y0': 472, 'y1': 523, 'new_text': 'WE HONOR', 'orig_cx': 783, 'fsize': 71, 'spacing': 11, 'orig_w': 517},
    {'name': 'big1',  'y0': 541, 'y1': 640, 'new_text': 'BRAVE',    'orig_cx': 829, 'fsize': 138, 'spacing': 198, 'orig_w': 1322},
    {'name': 'big2',  'y0': 656, 'y1': 755, 'new_text': 'LEGION',   'orig_cx': 863, 'fsize': 138, 'spacing': 65, 'orig_w': 917},
]

# Step 1: Build letter_mask (strict h >= 60% band_h) and chain_mask
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
        if h >= band_h * 0.6 and a >= 1000:
            # Letter: full-band-height CC
            letter_mask[y0+y_loc:y0+y_loc+h, x0:x0+w] = True
        else:
            # Chain: short CCs (w<12 or h<20)
            cc = (labels[y_loc:y_loc+h, x0:x0+w] == i)
            chain_mask[y0+y_loc:y0+y_loc+h, x0:x0+w] |= cc

print(f'[v319n] letter_mask px: {int(letter_mask.sum())}')
print(f'[v319n] chain_mask px: {int(chain_mask.sum())}')

# Step 2: Fill letter regions using scipy distance_transform_edt
# Source pixels = (NOT letter_mask) AND (NOT chain_mask)
source_pixels = ~letter_mask & ~chain_mask
print(f'[v319n] source_pixels px: {int(source_pixels.sum())}')

# Get nearest source pixel for each letter pixel
_, nearest_idx = distance_transform_edt(~source_pixels, return_distances=True, return_indices=True)
# nearest_idx shape: (2, H, W) where [0]=y, [1]=x
src_y = nearest_idx[0]
src_x = nearest_idx[1]

# Fill letter pixels with their nearest source pixel color
cleaned_rgb = img_rgb.copy()
cleaned_rgb[letter_mask] = img_rgb[src_y[letter_mask], src_x[letter_mask]]
print(f'[v319n] Distance transform fill done')

# Step 3: Anti-alias ring - 1-2px outside letter mask where gray is 80-160
# Get the "ring" zone: dilate letter mask 1px, intersect with gray 80-160
ring_mask = np.zeros((H, W), dtype=bool)
# Just outside the letter (1-2px dilate) but not in chain
for band in BANDS:
    y0, y1 = band['y0'], band['y1']
    band_letter = letter_mask[y0:y1, :]
    if not band_letter.any():
        continue
    # Dilate 1-2px (use 2x2 ellipse to be tight)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    band_dilated = cv2.dilate(band_letter.astype(np.uint8), k, iterations=1).astype(bool)
    band_ring = band_dilated & ~band_letter & ~chain_mask[y0:y1, :]
    # Only the anti-alias ring (gray 80-160)
    band_gray = gray[y0:y1, :]
    band_ring &= (band_gray >= 80) & (band_gray <= 160)
    ring_mask[y0:y1, :] |= band_ring

print(f'[v319n] anti-alias ring px: {int(ring_mask.sum())}')

# Apply cv2.inpaint NS radius=1 ONLY on the ring
if ring_mask.any():
    ring_uint8 = (ring_mask.astype(np.uint8) * 255)
    bgr = cv2.cvtColor(cleaned_rgb, cv2.COLOR_RGB2BGR)
    inpainted = cv2.inpaint(bgr, ring_uint8, 1, cv2.INPAINT_NS)
    cleaned_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
    print(f'[v319n] Anti-alias ring inpaint done')

# Save the cleaned base
Image.fromarray(cleaned_rgb).save(str(OUT_DIR / 'armed_lama_cleaned.png'))
print(f'[v319n] Saved cleaned base')

# Step 4: Write new text
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
    text_h = bbox[3] - bbox[1]
    n_chars = len(text)
    total_w = text_w + spacing * (n_chars - 1) if spacing > 0 else text_w
    paste_x = orig_cx - total_w // 2 - bbox[0]
    paste_y = y1_band - text_h - bbox[1]

    txt_layer = Image.new('RGBA', (total_w + 20, text_h + 20), (0, 0, 0, 0))
    txt_draw = ImageDraw.Draw(txt_layer)
    cur_x = 10 - bbox[0]
    for ch in text:
        txt_draw.text((cur_x, 10 - bbox[1]), ch, font=font, fill=(20, 20, 20, 255))
        ch_bbox = font.getbbox(ch)
        cur_x += (ch_bbox[2] - ch_bbox[0]) + spacing
    out_pil.paste(txt_layer, (paste_x, paste_y), txt_layer)
    print(f'[v319n] Wrote {band["name"]} "{text}" at ({paste_x},{paste_y}) w={total_w}')

OUT = OUT_DIR / 'armed_text_only_v319n.jpg'
out_pil.save(str(OUT), quality=95)
print(f'[v319n] Saved {OUT}')

# Compare
src_pil = Image.open(str(SRC)).convert('RGB')
combined = Image.new('RGB', (src_pil.width * 2 + 30, src_pil.height), (50, 50, 50))
combined.paste(src_pil, (0, 0))
combined.paste(out_pil.convert('RGB'), (src_pil.width + 30, 0))
combined.save(str(OUT_DIR / 'armed_compare_v319n.jpg'), quality=90)

src_crop = src_pil.crop((0, 440, W, 780))
res_crop = out_pil.crop((0, 440, W, 780))
band_compare = Image.new('RGB', (src_crop.width * 2 + 20, src_crop.height), (60, 60, 60))
band_compare.paste(src_crop, (0, 0))
band_compare.paste(res_crop, (src_crop.width + 20, 0))
band_compare.save(str(OUT_DIR / 'armed_text_band_compare_v319n.jpg'), quality=95)
print(f'[v319n] Saved compare images')
