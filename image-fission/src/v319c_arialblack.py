"""
v319c: Switch font from AbrilFatface (elegant Didone) to Arial Black (ultra-bold sans
with horizontal crossbars). This matches the visual style of the original ARMED/FORCES
much more closely than AbrilFatface.

Arial Black is sans-serif, but the original ARMED/FORCES reads as a heavy military
display typeface with very thick strokes and horizontal bars cutting through letters.
Arial Black matches that visual weight and presence (the user complaint '字的排版大小
差太多' was about WEIGHT, not serif/sans distinction).

Also: per-band font sizing matches cap height exactly.
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
OUT_DIR = str(PROJECT / "jobs" / "v319c")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
ARIAL_BLACK = r"C:/Windows/Fonts/ariblk.ttf"
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


# Find fsize that makes capH match each band height
def find_fsize_for_capH(target_capH, font_path):
    for fs in range(60, 250):
        font = ImageFont.truetype(font_path, fs)
        h_bbox = font.getbbox('H')
        capH = h_bbox[3] - h_bbox[1]
        if capH >= target_capH:
            return fs, font
    return 200, ImageFont.truetype(font_path, 200)


# Each band's text + target cap height
BANDS_DEF = [
    {'name': 'small', 'y0': 472, 'y1': 523, 'new_text': 'WE HONOR', 'orig_cx': 783, 'orig_w': 517},
    {'name': 'big1',  'y0': 541, 'y1': 640, 'new_text': 'BRAVE',   'orig_cx': 829, 'orig_w': 1322},
    {'name': 'big2',  'y0': 656, 'y1': 755, 'new_text': 'LEGION',  'orig_cx': 863, 'orig_w': 917},
]

img_bgr = load_jpg(SRC)
H, W = img_bgr.shape[:2]
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Step 1: text vs chain classification
chain_mask = np.zeros((H, W), dtype=np.uint8)
text_mask_global = np.zeros((H, W), dtype=bool)
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
        is_letter = (h >= band_h * 0.35)
        is_serif = (w >= 30) and (h >= 8) and (a >= 200) and (not is_letter) and (h < band_h * 0.5)
        is_text = is_letter or is_serif
        x0 = stats[i, cv2.CC_STAT_LEFT]
        y_loc = stats[i, cv2.CC_STAT_TOP]
        if is_text:
            text_mask_global[y0+y_loc:y0+y_loc+h, x0:x0+w] = True
        else:
            cc = (labels[y_loc:y_loc+h, x0:x0+w] == i)
            chain_mask[y0+y_loc:y0+y_loc+h, x0:x0+w][cc] = 255

# Step 2: tight mask dilate (2x2 ellipse)
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
text_mask_d = cv2.dilate(text_mask_global.astype(np.uint8) * 255, k, iterations=1) > 0
text_mask_d = text_mask_d & (chain_mask == 0)
print(f'[v319c] text mask px: {int(text_mask_d.sum())}')

mask_pil = Image.fromarray((text_mask_d.astype(np.uint8) * 255), mode='L')
src_pil = Image.fromarray(img_rgb)
print('[v319c] Running LaMa inpaint...')
cleaned = lama_inpaint(src_pil, mask_pil, removal_strength=235, edge_smoothness=3)
cleaned.save(Path(OUT_DIR) / 'armed_lama_cleaned.png', quality=95)

# Step 3: per-band Arial Black sized to capH = band height
out_pil = cleaned.copy()
draw = ImageDraw.Draw(out_pil)

BANDS = []
for b in BANDS_DEF:
    band_h = b['y1'] - b['y0']
    fsize, font = find_fsize_for_capH(band_h, ARIAL_BLACK)
    h_bbox = font.getbbox('H')
    capH = h_bbox[3] - h_bbox[1]
    text = b['new_text']
    bbox = font.getbbox(text)
    nat_w = bbox[2] - bbox[0]
    # Tighten horizontal: nat width is wider than original; we want total ≈ orig_w
    # Use letter spacing to fill
    desired_w = b['orig_w']
    spacing = max(0, (desired_w - nat_w) // max(len(text) - 1, 1))
    total_w = nat_w + spacing * (len(text) - 1)
    paste_x = b['orig_cx'] - total_w // 2 - bbox[0]
    paste_y = b['y1'] - bbox[3]
    BANDS.append({**b, 'fsize': fsize, 'font': font, 'bbox': bbox, 'nat_w': nat_w,
                  'spacing': spacing, 'total_w': total_w,
                  'paste_x': paste_x, 'paste_y': paste_y, 'capH': capH})
    print(f'[v319c] {b["name"]}: capH={capH} fsize={fsize} "{text}" nat_w={nat_w} spacing={spacing} '
          f'total_w={total_w} paste=({paste_x},{paste_y}) ink_y={paste_y + bbox[1]}..{paste_y + bbox[3]}')

for band in BANDS:
    draw.text((band['paste_x'], band['paste_y']),
              band['new_text'],
              font=band['font'],
              anchor='lt',
              fill=(20, 20, 20),
              spacing=band['spacing'])

OUT = Path(OUT_DIR) / 'armed_text_only_v319c.jpg'
out_pil.save(OUT, quality=95)

src_compare = Image.open(SRC).convert('RGB')
combined = Image.new('RGB', (src_compare.width * 2 + 30, src_compare.height), (50, 50, 50))
combined.paste(src_compare, (0, 0))
combined.paste(out_pil, (src_compare.width + 30, 0))
combined.save(Path(OUT_DIR) / 'armed_compare_v319c.jpg', quality=90)

src_crop = src_compare.crop((0, 440, 1556, 780))
res_crop = out_pil.crop((0, 440, 1556, 780))
band_compare = Image.new('RGB', (src_crop.width * 2 + 20, src_crop.height), (60, 60, 60))
band_compare.paste(src_crop, (0, 0))
band_compare.paste(res_crop, (src_crop.width + 20, 0))
band_compare.save(Path(OUT_DIR) / 'armed_text_band_compare_v319c.jpg', quality=95)
print(f'[v319c] Saved {OUT}')