"""v319c2: v319c's pipeline but with v316e's STRICT mask classification.

The previous "LaMa haze" in v319c was actually caused by lax mask (h>=0.35*band_h)
including dark camo patches as text. LaMa then attempted to "rebuild" those
non-text dark patches and left halos around them. v319c2 uses the strict
v316e classification (h>=0.6 for letter, w>=30+h>=8+a>=200 for serif)
that correctly separates text from camo.

Same as v319c but with strict mask, 2x2 dilate, Arial Black, no chain logic
issues.
"""
import cv2
import numpy as np
import shutil
import tempfile
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

PROJECT = Path("E:/Desktop/双接口/image-fission")
SRC = str(PROJECT / "jobs" / "v315" / "source_armed.jpg")
OUT_DIR = str(PROJECT / "jobs" / "v319c2")
LAMA_PT = str(PROJECT / "ComfyUI" / "models" / "RMBG" / "Lama" / "big-lama.pt")
ARIAL_BLACK = r"C:/Windows/Fonts/ariblk.ttf"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

def load_jpg(p):
    with open(p, 'rb') as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def load_lama(ckpt_path):
    ascii_dir = tempfile.mkdtemp(prefix="lama_")
    ascii_ckpt = Path(ascii_dir) / "big-lama.pt"
    shutil.copy(ckpt_path, ascii_ckpt)
    model = torch.jit.load(str(ascii_ckpt), map_location='cpu')
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, ascii_dir

def pad8(img):
    w, h = img.size
    nw, nh = ((w + 7) // 8) * 8, ((h + 7) // 8) * 8
    canvas = Image.new(img.mode, (nw, nh), 0)
    canvas.paste(img, (0, 0))
    return canvas

img_bgr = load_jpg(SRC)
H, W = img_bgr.shape[:2]
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

BANDS_DEF = [
    {'name': 'small', 'y0': 472, 'y1': 523, 'new_text': 'WE HONOR', 'orig_cx': 783, 'orig_w': 517},
    {'name': 'big1',  'y0': 541, 'y1': 640, 'new_text': 'BRAVE',   'orig_cx': 829, 'orig_w': 1322},
    {'name': 'big2',  'y0': 656, 'y1': 755, 'new_text': 'LEGION',  'orig_cx': 863, 'orig_w': 917},
]

# STRICT v316e classification
chain_mask = np.zeros((H, W), dtype=np.uint8)
letter_mask = np.zeros((H, W), dtype=bool)
serif_mask = np.zeros((H, W), dtype=bool)
for band in BANDS_DEF:
    y0, y1 = band['y0'], band['y1']
    band_h = y1 - y0
    band_gray = gray[y0:y1, :]
    dark = (band_gray < 80).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    for i in range(1, n):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        a = stats[i, cv2.CC_STAT_AREA]
        is_letter = (h >= band_h * 0.6) and (a >= 1000)
        is_serif = (not is_letter) and (w >= 30) and (h >= 8) and (h < band_h * 0.5) and (a >= 200)
        x0 = stats[i, cv2.CC_STAT_LEFT]
        y_loc = stats[i, cv2.CC_STAT_TOP]
        if is_letter:
            letter_mask[y0+y_loc:y0+y_loc+h, x0:x0+w] = True
        elif is_serif:
            serif_mask[y0+y_loc:y0+y_loc+h, x0:x0+w] = True
        else:
            cc = (labels[y_loc:y_loc+h, x0:x0+w] == i)
            chain_mask[y0+y_loc:y0+y_loc+h, x0:x0+w][cc] = 255

text_mask = (letter_mask | serif_mask) & (chain_mask == 0)
# 2x2 dilate (v319c's choice to capture anti-aliased edges)
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
text_mask_d = cv2.dilate(text_mask.astype(np.uint8) * 255, k, iterations=1) > 0
text_mask_d = text_mask_d & (chain_mask == 0)
print(f'[v319c2] text mask: {int(text_mask.sum())}, after 2x2 dilate: {int(text_mask_d.sum())}')

# Save mask
Image.fromarray((text_mask_d.astype(np.uint8) * 255)).save(str(Path(OUT_DIR) / 'text_mask.png'))

# Run LaMa with v268 pipeline
print(f'[v319c2] Loading LaMa...')
model, _ = load_lama(LAMA_PT)
src_pil = Image.fromarray(img_rgb)
mask_pil = Image.fromarray((text_mask_d.astype(np.uint8) * 255).astype(np.uint8), mode='L')

# v268 pipeline
w, h = src_pil.size
p_img = pad8(src_pil)
p_mask = pad8(mask_pil)
p_mask = ImageOps.invert(p_mask)
p_mask = p_mask.filter(ImageFilter.GaussianBlur(radius=8))
p_mask = p_mask.point(lambda x: 0 if x > 230 else 255)

dev = "cuda" if torch.cuda.is_available() else "cpu"
img_t = torch.from_numpy(np.array(p_img).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
mask_t = torch.from_numpy(np.array(p_mask).astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(dev)
print(f'[v319c2] Running LaMa...')
with torch.inference_mode():
    res = model(img_t, mask_t)
out = Image.fromarray((res[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
if out.width > w or out.height > h:
    out = out.crop((0, 0, w, h))
out.save(str(Path(OUT_DIR) / 'armed_lama_cleaned.png'))

# Copy back chain
cleaned_rgb = np.array(out.convert('RGB'))
copy_back = (chain_mask > 0)
cleaned_rgb[copy_back] = img_rgb[copy_back]
cleaned_pil = Image.fromarray(cleaned_rgb)
cleaned_pil.save(str(Path(OUT_DIR) / 'armed_lama_cleaned_with_chain.png'))

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
    text = b['new_text']
    bbox = font.getbbox(text)
    nat_w = bbox[2] - bbox[0]
    spacing = max(0, (b['orig_w'] - nat_w) // max(len(text) - 1, 1))
    total_w = nat_w + spacing * (len(text) - 1)
    paste_x = b['orig_cx'] - total_w // 2 - bbox[0]
    paste_y = b['y1'] - bbox[3]
    draw.text((paste_x, paste_y), text, font=font, anchor='lt', fill=(20, 20, 20), spacing=spacing)
    print(f'[v319c2] {b["name"]}: fsize={fsize} "{text}" paste=({paste_x},{paste_y})')

OUT = Path(OUT_DIR) / 'armed_text_only_v319c2.jpg'
out_pil.save(str(OUT), quality=95)

src_compare = Image.open(SRC).convert('RGB')
combined = Image.new('RGB', (src_compare.width * 2 + 30, src_compare.height), (50, 50, 50))
combined.paste(src_compare, (0, 0))
combined.paste(out_pil, (src_compare.width + 30, 0))
combined.save(str(Path(OUT_DIR) / 'armed_compare_v319c2.jpg'), quality=90)

src_crop = src_compare.crop((0, 440, 1556, 780))
res_crop = out_pil.crop((0, 440, 1556, 780))
band_compare = Image.new('RGB', (src_crop.width * 2 + 20, src_crop.height), (60, 60, 60))
band_compare.paste(src_crop, (0, 0))
band_compare.paste(res_crop, (src_crop.width + 20, 0))
band_compare.save(str(Path(OUT_DIR) / 'armed_text_band_compare_v319c2.jpg'), quality=95)
print(f'[v319c2] Saved {OUT}')
