"""
v315_text_only_chain_preserve.py
Pure text-only fission for ARMED-style no-main-subject images.
Fixes v314b problems:
  1. Dog tag chain was eaten by cv2.inpaint (CC area filter was too loose)
  2. Original text residual (e.g. 'FORCES' under 'LEGION') was visible
  3. New text didn't match original ARMED/FORCES letter height or position

Strategy:
  - NO mode3 diffusion (chain + dog tag must survive — mode3 keeps breaking them)
  - Surgical CC mask: only include wide+thick CCs as text mask
    → chain links (small CCs, h<=15) are excluded from inpaint
  - NS inpaint with moderate radius for clean text removal
  - Restore chain pixels from source if any got damaged
  - Anton font sized to match original letter height in each band
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC = r'E:/Desktop/双接口/image-fission/jobs/v315/source_armed.jpg'
OUT = r'E:/Desktop/双接口/image-fission/jobs/v315/armed_text_only_v315.jpg'
COMPARE = r'E:/Desktop/双接口/image-fission/jobs/v315/armed_compare_v315.jpg'
FONT_PATH = r'E:/Desktop/双接口/image-fission/ComfyUI/models/fonts/Anton-Regular.ttf'

# ---- Load (cv2.imdecode handles unicode paths reliably) ----
with open(SRC, 'rb') as f:
    data = f.read()
arr = np.frombuffer(data, dtype=np.uint8)
img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
H, W = img_bgr.shape[:2]
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
print(f'[v315] Loaded {W}x{H}')

# ---- Measured text band positions (from native source 1556x2000) ----
# WE SUPPORT THE band: y=472..523, h=51
# ARMED band:           y=541..640, h=99
# FORCES band:          y=656..755, h=99
# (these are the three text bands; dog tag starts below ~y=720)
BANDS = [
    {'name': 'small', 'y0': 472, 'y1': 523, 'min_h': 25, 'min_w': 12, 'min_area': 200},
    {'name': 'big1',  'y0': 541, 'y1': 640, 'min_h': 60, 'min_w': 30, 'min_area': 1000},
    {'name': 'big2',  'y0': 656, 'y1': 755, 'min_h': 60, 'min_w': 30, 'min_area': 1000},
]

# ---- Build text-only mask (exclude chain links) ----
mask = np.zeros((H, W), dtype=np.uint8)
chain_preserve_mask = np.zeros((H, W), dtype=np.uint8)  # pixels to restore after inpaint

for band in BANDS:
    y0, y1 = band['y0'], band['y1']
    band_gray = gray[y0:y1, :]
    text_pix = (band_gray < 75).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(text_pix, connectivity=8)
    band_mask = np.zeros_like(text_pix)
    for i in range(1, n):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        a = stats[i, cv2.CC_STAT_AREA]
        # Text characters are tall+wide; chain links are short (h<=15) and small
        is_text = (w >= band['min_w']) and (h >= band['min_h']) and (a >= band['min_area'])
        if is_text:
            band_mask[labels == i] = 255
        else:
            # Record chain pixels to preserve
            chain_preserve_mask[y0 + y : y0 + y + h, x : x + w][labels[y:y+h, x:x+w] == i] = 255
    mask[y0:y1, :][band_mask > 0] = 255

# Also: small dark specs that look like AI-generated noise (small inpaint leftovers).
# Dilate the mask slightly so anti-alias edges of letters are also cleaned.
mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
print(f'[v315] Text mask pixels: {int((mask > 0).sum())}')
print(f'[v315] Chain preserve pixels: {int((chain_preserve_mask > 0).sum())}')

# ---- Save mask for inspection ----
cv2.imwrite(r'E:/Desktop/双接口/image-fission/jobs/v315/mask_text.png', mask)
cv2.imwrite(r'E:/Desktop/双接口/image-fission/jobs/v315/mask_chain.png', chain_preserve_mask)

# ---- Inpaint (NS works better for text on textured backgrounds than TELEA) ----
inpainted = cv2.inpaint(img_bgr, mask, 4, cv2.INPAINT_NS)
print('[v315] Inpaint done')

# ---- Restore any chain pixels that got damaged by inpaint bleed ----
# The chain should be preserved by virtue of being excluded from mask, but be safe.
if chain_preserve_mask.sum() > 0:
    # Where original was dark AND we marked as chain AND inpaint changed it, restore original
    changed = (cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY) > 100) & (chain_preserve_mask > 0) & (gray < 100)
    inpainted[changed] = img_bgr[changed]
    print(f'[v315] Restored {int(changed.sum())} chain pixels from source')

# ---- Write new text at original positions ----
inpainted_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
out_pil = Image.fromarray(inpainted_rgb)
draw = ImageDraw.Draw(out_pil)

# Font sizes calibrated to match original text height:
#   big bands h=99 → Anton font size 138 (Anton cap-height ≈ 0.72 × size, gives ~99px cap)
#   small band h=51 → Anton font size 72
FONT_SIZES = {'small': 72, 'big1': 138, 'big2': 138}
NEW_TEXTS = {'small': 'WE HONOR OUR HEROES', 'big1': 'BRAVE', 'big2': 'LEGION'}

# Compute original text X-extent for each band (use source mask)
def measure_text_x_extent(gray, band_y0, band_y1, min_h, min_w, min_area):
    """Find leftmost/rightmost text pixel in this band."""
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
    return min(xs), max(xs)

# Place text in each band
for band in BANDS:
    text = NEW_TEXTS[band['name']]
    fsize = FONT_SIZES[band['name']]
    font = ImageFont.truetype(FONT_PATH, fsize)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    # Measure original X-extent to match original layout
    orig_x0, orig_x1 = measure_text_x_extent(
        gray, band['y0'], band['y1'],
        band['min_w'], band['min_h'], band['min_area'],
    )
    orig_cx = (orig_x0 + orig_x1) // 2

    # X: place new text centered at original center
    x = orig_cx - tw // 2 - bbox[0]
    # Y: vertically center within band
    y_band_center = (band['y0'] + band['y1']) // 2
    y = y_band_center - th // 2 - bbox[1]

    draw.text((x, y), text, font=font, fill=(20, 20, 20))
    print(f'[v315] Band {band["name"]}: text="{text}" font={fsize}px '
          f'pos=({x},{y}) orig_x=({orig_x0},{orig_x1}) orig_cx={orig_cx}')

out_pil.save(OUT, quality=95)
print(f'[v315] Saved {OUT}')

# ---- Side-by-side compare: source | result ----
src_pil = Image.open(SRC).convert('RGB')
W2 = src_pil.width
H2 = src_pil.height
combined = Image.new('RGB', (W2 * 2 + 30, H2), (50, 50, 50))
combined.paste(src_pil, (0, 0))
combined.paste(out_pil, (W2 + 30, 0))
combined.save(COMPARE, quality=90)
print(f'[v315] Saved compare {COMPARE}')