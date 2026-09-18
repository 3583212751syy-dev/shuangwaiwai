"""v319h compose: take SDXL inpaint result, copy-back chain/dog-tag, write new text."""
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path("E:/Desktop/双接口/image-fission")
SRC = str(PROJECT / "jobs" / "v315" / "source_armed.jpg")
CLEANED = str(PROJECT / "jobs" / "v319h" / "armed_sdxl_cleaned_raw.png")
OUT_DIR = str(PROJECT / "jobs" / "v319h")
ARIAL_BLACK = r"C:/Windows/Fonts/ariblk.ttf"

def load_jpg(p):
    with open(p, 'rb') as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

img_bgr = load_jpg(SRC)
H, W = img_bgr.shape[:2]
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

BANDS_DEF = [
    {'name': 'small', 'y0': 472, 'y1': 523, 'new_text': 'WE HONOR', 'orig_cx': 783, 'orig_w': 517},
    {'name': 'big1',  'y0': 541, 'y1': 640, 'new_text': 'BRAVE',   'orig_cx': 829, 'orig_w': 1322},
    {'name': 'big2',  'y0': 656, 'y1': 755, 'new_text': 'LEGION',  'orig_cx': 863, 'orig_w': 917},
]

# Recompute chain + dog-tag masks (same as v319h)
chain_mask = np.zeros((H, W), dtype=np.uint8)
letter_mask = np.zeros((H, W), dtype=bool)
for band in BANDS_DEF:
    y0, y1 = band['y0'], band['y1']
    band_h = y1 - y0
    band_gray = gray[y0:y1, :]
    dark = (band_gray < 80).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    for i in range(1, n):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        is_letter = (h >= band_h * 0.35)
        x0 = stats[i, cv2.CC_STAT_LEFT]
        y_loc = stats[i, cv2.CC_STAT_TOP]
        if is_letter:
            letter_mask[y0+y_loc:y0+y_loc+h, x0:x0+w] = True
        else:
            cc = (labels[y_loc:y_loc+h, x0:x0+w] == i)
            chain_mask[y0+y_loc:y0+y_loc+h, x0:x0+w][cc] = 255

dog_tag_mask = np.zeros((H, W), dtype=np.uint8)
for band in BANDS_DEF:
    y0, y1 = band['y0'], band['y1']
    band_gray = gray[y0:y1, :]
    med_dark = ((band_gray >= 80) & (band_gray < 180)).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(med_dark, connectivity=8)
    for i in range(1, n):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        a = stats[i, cv2.CC_STAT_AREA]
        if 8000 < a < 30000 and h > 80 and 0.5 < w/h < 1.5:
            x0 = stats[i, cv2.CC_STAT_LEFT]
            y_loc = stats[i, cv2.CC_STAT_TOP]
            cc = (labels[y_loc:y_loc+h, x0:x0+w] == i)
            dog_tag_mask[y0+y_loc:y0+y_loc+h, x0:x0+w][cc] = 255

# Load SDXL cleaned result
cleaned_pil = Image.open(CLEANED).convert('RGB')
if cleaned_pil.size != (W, H):
    cleaned_pil = cleaned_pil.resize((W, H))
cleaned_rgb = np.array(cleaned_pil)

# Copy back chain + dog tag from source
copy_back = (chain_mask > 0) | (dog_tag_mask > 0)
cleaned_rgb[copy_back] = img_rgb[copy_back]
cleaned_pil = Image.fromarray(cleaned_rgb)
cleaned_pil.save(str(Path(OUT_DIR) / 'armed_sdxl_cleaned.png'))
print(f'[compose] copied back chain+dogtag px: {int(copy_back.sum())}')

# Write new text
out_pil = cleaned_pil.copy()
draw = ImageDraw.Draw(out_pil)

def find_fsize_for_capH(target_capH, font_path):
    for fs in range(60, 250):
        font = ImageFont.truetype(font_path, fs)
        h_bbox = font.getbbox('H')
        capH = h_bbox[3] - h_bbox[1]
        if capH >= target_capH:
            return fs, font
    return 200, ImageFont.truetype(font_path, 200)

for b in BANDS_DEF:
    band_h = b['y1'] - b['y0']
    fsize, font = find_fsize_for_capH(band_h, ARIAL_BLACK)
    h_bbox = font.getbbox('H')
    capH = h_bbox[3] - h_bbox[1]
    text = b['new_text']
    bbox = font.getbbox(text)
    nat_w = bbox[2] - bbox[0]
    spacing = max(0, (b['orig_w'] - nat_w) // max(len(text) - 1, 1))
    total_w = nat_w + spacing * (len(text) - 1)
    paste_x = b['orig_cx'] - total_w // 2 - bbox[0]
    paste_y = b['y1'] - bbox[3]
    draw.text((paste_x, paste_y), text, font=font, anchor='lt', fill=(20, 20, 20), spacing=spacing)
    print(f'[compose] {b["name"]}: capH={capH} fsize={fsize} "{text}" paste=({paste_x},{paste_y}) total_w={total_w}')

OUT = Path(OUT_DIR) / 'armed_text_only_v319h.jpg'
out_pil.save(str(OUT), quality=95)

src_compare = Image.open(SRC).convert('RGB')
combined = Image.new('RGB', (src_compare.width * 2 + 30, src_compare.height), (50, 50, 50))
combined.paste(src_compare, (0, 0))
combined.paste(out_pil, (src_compare.width + 30, 0))
combined.save(str(Path(OUT_DIR) / 'armed_compare_v319h.jpg'), quality=90)

src_crop = src_compare.crop((0, 440, 1556, 780))
res_crop = out_pil.crop((0, 440, 1556, 780))
band_compare = Image.new('RGB', (src_crop.width * 2 + 20, src_crop.height), (60, 60, 60))
band_compare.paste(src_crop, (0, 0))
band_compare.paste(res_crop, (src_crop.width + 20, 0))
band_compare.save(str(Path(OUT_DIR) / 'armed_text_band_compare_v319h.jpg'), quality=95)
print(f'[compose] Saved {OUT}')
