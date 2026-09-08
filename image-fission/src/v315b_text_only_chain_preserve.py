"""
v315b_text_only_chain_preserve.py
Text-only fission with full-band mask for clean removal + Impact font matching
ARMED/FORCES bold style.

Fixes over v314b:
  - Chain was missing: surgical CC mask excluded chain pixels, but cv2.inpaint NS
    on high-freq camo left text residue (shadow of "AR", "ED", "FO", "RCES")
  - Text wasn't fully cleaned: only pixel-level text mask, not band-wide

Fixes over v315:
  - Mask = full band rectangle minus chain CCs (not just text pixels)
  - NS inpaint on the FULL band → fully clean background, no residue
  - Chain still preserved (excluded from mask by CC filter)
  - Impact font instead of Anton → matches bold ARMED/FORCES style

Font note:
  - Original ARMED/FORCES are heavy bold blocky military sans-serif
  - Impact.ttf (Windows system) is closest available match
  - DINNextLTPro-Bold could also work but Impact is more condensed
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC = r'E:/Desktop/双接口/image-fission/jobs/v315/source_armed.jpg'
OUT = r'E:/Desktop/双接口/image-fission/jobs/v315/armed_text_only_v315b.jpg'
COMPARE = r'E:/Desktop/双接口/image-fission/jobs/v315/armed_compare_v315b.jpg'
MASK_OUT = r'E:/Desktop/双接口/image-fission/jobs/v315/mask_v315b.png'
FONT_PATH = r'C:/Windows/Fonts/impact.ttf'

# ---- Load (cv2.imdecode handles unicode paths reliably) ----
with open(SRC, 'rb') as f:
    data = f.read()
arr = np.frombuffer(data, dtype=np.uint8)
img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
H, W = img_bgr.shape[:2]
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
print(f'[v315b] Loaded {W}x{H}')

# ---- Measured text band positions (from native 1556x2000 source) ----
BANDS = [
    {'name': 'small', 'y0': 472, 'y1': 523, 'min_h_chain': 25, 'min_w_chain': 12, 'min_area_chain': 200, 'x_pad': 30},
    {'name': 'big1',  'y0': 541, 'y1': 640, 'min_h_chain': 60, 'min_w_chain': 30, 'min_area_chain': 1000, 'x_pad': 30},
    {'name': 'big2',  'y0': 656, 'y1': 755, 'min_h_chain': 60, 'min_w_chain': 30, 'min_area_chain': 1000, 'x_pad': 30},
]

# ---- Build FULL-band mask, with chain CCs subtracted ----
mask = np.zeros((H, W), dtype=np.uint8)

for band in BANDS:
    y0, y1 = band['y0'], band['y1']
    pad = band['x_pad']

    # 1. Start with full band rectangle (padded slightly)
    band_mask = np.zeros((H, W), dtype=np.uint8)
    band_mask[y0:y1, pad:W-pad] = 255

    # 2. Identify chain CCs within the band (small dark components that aren't text)
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
        # If small (chain link / thin curve), subtract from mask (preserve)
        is_chain = not (w >= band['min_w_chain'] and h >= band['min_h_chain'] and a >= band['min_area_chain'])
        if is_chain:
            chain_subtract[y0 + y_local : y0 + y_local + h, pad + x_local : pad + x_local + w][labels[y_local:y_local+h, x_local:x_local+w] == i] = 255

    # 3. Final band mask = band rectangle MINUS chain
    band_mask_final = band_mask.copy()
    band_mask_final[chain_subtract > 0] = 0

    # 4. Merge into global mask
    mask = np.maximum(mask, band_mask_final)
    print(f'[v315b] Band {band["name"]} y={y0}..{y1}: mask={int((band_mask>0).sum())} chain_subtract={int((chain_subtract>0).sum())}')

print(f'[v315b] Total mask pixels: {int((mask > 0).sum())}')
cv2.imwrite(MASK_OUT, mask)

# ---- Inpaint: NS with radius 4 ----
inpainted = cv2.inpaint(img_bgr, mask, 4, cv2.INPAINT_NS)
print('[v315b] Inpaint NS done')

# Also try TELEA for comparison (kept for inspection if needed)
inpainted_telea = cv2.inpaint(img_bgr, mask, 4, cv2.INPAINT_TELEA)
cv2.imwrite(r'E:/Desktop/双接口/image-fission/jobs/v315/inpainted_NS.png', inpainted)
cv2.imwrite(r'E:/Desktop/双接口/image-fission/jobs/v315/inpainted_TELEA.png', inpainted_telea)

# ---- Write new text at original positions ----
inpainted_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
out_pil = Image.fromarray(inpainted_rgb)
draw = ImageDraw.Draw(out_pil)

# Impact font sizing: cap height ≈ 0.71 × font_size
# Big band h=99 → font_size ≈ 138 (cap height ~99)
# Small band h=51 → font_size ≈ 72 (cap height ~51)
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
    print(f'[v315b] Band {band["name"]}: text="{text}" font={fsize}px '
          f'pos=({x},{y}) orig_x=({orig_x0},{orig_x1}) orig_cx={orig_cx}')

out_pil.save(OUT, quality=95)
print(f'[v315b] Saved {OUT}')

# ---- Side-by-side compare: source | NS result ----
src_pil = Image.open(SRC).convert('RGB')
W2 = src_pil.width
H2 = src_pil.height
combined = Image.new('RGB', (W2 * 2 + 30, H2), (50, 50, 50))
combined.paste(src_pil, (0, 0))
combined.paste(out_pil, (W2 + 30, 0))
combined.save(COMPARE, quality=90)
print(f'[v315b] Saved compare {COMPARE}')