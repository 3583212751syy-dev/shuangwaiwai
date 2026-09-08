"""
v318c: LaMa fill + Anton natural-width + letter spacing. v318b's sx=2.0
stretched Anton into blobs (BRAVE -> RRAVF). Use letter spacing instead
of horizontal stretch to widen the text without distorting letter shape.
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
OUT_DIR = str(PROJECT / "jobs" / "v318")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
ANTON_PATH = str(PROJECT / "ComfyUI" / "models" / "fonts" / "Anton-Regular.ttf")
MODEL_PATH = PROJECT / "ComfyUI" / "models" / "RMBG" / "Lama" / "big-lama.pt"

_LAMA = None


def load_lama():
    global _LAMA
    if _LAMA is not None:
        return _LAMA
    src = str(MODEL_PATH)
    if any(ord(c) > 127 for c in src):
        tmp = Path(tempfile.gettempdir()) / "big-lama.pt"
        if not tmp.exists() or tmp.stat().st_size != Path(src).stat().st_size:
            shutil.copyfile(src, tmp)
        src = str(tmp)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(src, map_location=dev)
    model.eval().to(dev)
    _LAMA = (model, dev)
    return _LAMA


def pad_image(image, is_mask=False):
    w, h = image.size
    if w % 8 != 0:
        w += 8 - w % 8
    if h % 8 != 0:
        h += 8 - h % 8
    fill = 0 if is_mask else None
    padded = Image.new(image.mode, (w, h), color=fill)
    padded.paste(image, (0, 0))
    return padded


def lama_inpaint(src_pil, mask_pil, removal_strength=230, edge_smoothness=8):
    model, dev = load_lama()
    w, h = src_pil.size
    p_img = pad_image(src_pil)
    p_mask = pad_image(mask_pil, is_mask=True)
    if p_mask.size != p_img.size:
        p_mask = p_mask.resize(p_img.size, Image.LANCZOS)
    p_mask = ImageOps.invert(p_mask)
    p_mask = p_mask.filter(ImageFilter.GaussianBlur(radius=edge_smoothness))
    gray = p_mask.point(lambda x: 0 if x > removal_strength else 255)
    img_t = torch.from_numpy(np.array(p_img).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
    mask_t = torch.from_numpy(np.array(gray).astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(dev)
    with torch.inference_mode():
        res = model(img_t, mask_t)
    res_img = Image.fromarray((res[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
    if res_img.width > w or res_img.height > h:
        res_img = res_img.crop((0, 0, w, h))
    return res_img.convert("RGB")


def load_jpg(p):
    with open(p, 'rb') as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# BANDS: original X center, target text, letter spacing (px between letters)
# Original widths: small 517, big1 1322, big2 917
# Anton natural widths: HEROES ~161, BRAVE ~265, LEGION ~284
# To roughly hit the original width, spacing ~ (orig_w - nat_w) / (n-1)
#   small: (517-161)/5  = 71
#   big1:  (1322-265)/4 = 264
#   big2:  (917-284)/5  = 127
# Those would be huge. Cap at 24 to keep readability (military stencil feel).
BANDS = [
    {'name': 'small', 'y0': 472, 'y1': 523, 'new_text': 'HEROES', 'orig_cx': 783, 'spacing': 14},
    {'name': 'big1',  'y0': 541, 'y1': 640, 'new_text': 'BRAVE',  'orig_cx': 829, 'spacing': 24},
    {'name': 'big2',  'y0': 656, 'y1': 755, 'new_text': 'LEGION', 'orig_cx': 863, 'spacing': 18},
]

img_bgr = load_jpg(SRC)
H, W = img_bgr.shape[:2]
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Step 1: text vs chain classification (height-based, from v316e — correct)
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
        is_letter = (h >= band_h * 0.35)
        is_serif = (w >= 30) and (h >= 20) and (a >= 1500) and (not is_letter)
        is_text = is_letter or is_serif
        x0 = stats[i, cv2.CC_STAT_LEFT]
        y_loc = stats[i, cv2.CC_STAT_TOP]
        if is_text:
            text_mask_global[y0+y_loc:y0+y_loc+h, x0:x0+w] = True
        else:
            cc = (labels[y_loc:y_loc+h, x0:x0+w] == i)
            chain_mask[y0+y_loc:y0+y_loc+h, x0:x0+w][cc] = 255

# Step 2: dilate + LaMa
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
text_mask_d = cv2.dilate(text_mask_global.astype(np.uint8) * 255, k, iterations=1) > 0
text_mask_d = text_mask_d & (chain_mask == 0)
mask_pil = Image.fromarray((text_mask_d.astype(np.uint8) * 255), mode='L')
print(f'[v318c] text mask px: {int(text_mask_d.sum())}')

src_pil = Image.fromarray(img_rgb)
print('[v318c] Running LaMa inpaint...')
cleaned = lama_inpaint(src_pil, mask_pil, removal_strength=235, edge_smoothness=6)
cleaned.save(Path(OUT_DIR) / 'armed_lama_cleaned.png', quality=95)

# Step 3: write text with letter spacing
out_pil = cleaned.copy()
draw = ImageDraw.Draw(out_pil)
for band in BANDS:
    text = band['new_text']
    target_h = band['y1'] - band['y0']
    fsize = 50
    for fs in range(50, 250):
        font = ImageFont.truetype(ANTON_PATH, fs)
        bbox = draw.textbbox((0, 0), text, font=font, anchor='lt')
        nat_h = bbox[3] - bbox[1]
        if nat_h >= target_h:
            fsize = fs
            break
    bbox = draw.textbbox((0, 0), text, font=font, anchor='lt')
    nat_w = bbox[2] - bbox[0]
    nat_h = bbox[3] - bbox[1]
    spacing = band['spacing']
    total_w = nat_w + spacing * (len(text) - 1)
    paste_x = band['orig_cx'] - total_w // 2 - bbox[0]
    paste_y = (band['y0'] + band['y1']) // 2 - nat_h // 2 - bbox[1]
    # PIL draw.text supports 'spacing' kwarg (PIL >= 8.0)
    draw.text((paste_x, paste_y), text, font=font, anchor='lt',
              fill=(20, 20, 20), spacing=spacing)
    print(f'[v318c] {band["name"]}: "{text}" fsize={fsize} spacing={spacing} total_w={total_w} paste=({paste_x},{paste_y})')

OUT = Path(OUT_DIR) / 'armed_text_only_v318c.jpg'
out_pil.save(OUT, quality=95)

src_compare = Image.open(SRC).convert('RGB')
combined = Image.new('RGB', (src_compare.width * 2 + 30, src_compare.height), (50, 50, 50))
combined.paste(src_compare, (0, 0))
combined.paste(out_pil, (src_compare.width + 30, 0))
combined.save(Path(OUT_DIR) / 'armed_compare_v318c.jpg', quality=90)

src_crop = src_compare.crop((0, 440, 1556, 780))
res_crop = out_pil.crop((0, 440, 1556, 780))
band_compare = Image.new('RGB', (src_crop.width * 2 + 20, src_crop.height), (60, 60, 60))
band_compare.paste(src_crop, (0, 0))
band_compare.paste(res_crop, (src_crop.width + 20, 0))
band_compare.save(Path(OUT_DIR) / 'armed_text_band_compare_v318c.jpg', quality=95)
print(f'[v318c] Saved {OUT}')
