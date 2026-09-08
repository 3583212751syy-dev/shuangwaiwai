"""v319f: LaMa mask = letter CC intersected with gray<80 (DEEP black cores only).
This is the tightest possible mask — only the letter body interiors, no
anti-aliased edges, no surrounding context.

Why v319e still had haze bands:
  Even with no dilate and edge_smoothness=1, LaMa still processes the 1-2px
  mask boundary (because removal_strength cuts off soft pixels). The boundary
  'leaks' into surrounding pixels via the LaMa Fourier conv.

v319f fix:
  mask = (letter CC bodies) & (gray < 80)  — only the dark letter cores
  Anti-aliased edges (80 <= gray < 160) are NOT in mask, so LaMa doesn't
  touch them; they remain as natural anti-alias in the output.
  Chain pixels + dog tag pixels are not in mask, never touched.
  Result: text removed but NO haze band; only the dark cores get filled.
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
OUT_DIR = str(PROJECT / "jobs" / "v319f")
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


def find_fsize_for_capH(target_capH, font_path):
    for fs in range(60, 250):
        font = ImageFont.truetype(font_path, fs)
        h_bbox = font.getbbox('H')
        capH = h_bbox[3] - h_bbox[1]
        if capH >= target_capH:
            return fs, font
    return 200, ImageFont.truetype(font_path, 200)


BANDS_DEF = [
    {'name': 'small', 'y0': 472, 'y1': 523, 'new_text': 'WE HONOR', 'orig_cx': 783, 'orig_w': 517},
    {'name': 'big1',  'y0': 541, 'y1': 640, 'new_text': 'BRAVE',   'orig_cx': 829, 'orig_w': 1322},
    {'name': 'big2',  'y0': 656, 'y1': 755, 'new_text': 'LEGION',  'orig_cx': 863, 'orig_w': 917},
]

img_bgr = load_jpg(SRC)
H, W = img_bgr.shape[:2]
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Step 1: letter CC mask (no is_serif, only letter body)
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
        a = stats[i, cv2.CC_STAT_AREA]
        is_letter = (h >= band_h * 0.35)
        x0 = stats[i, cv2.CC_STAT_LEFT]
        y_loc = stats[i, cv2.CC_STAT_TOP]
        if is_letter:
            letter_mask[y0+y_loc:y0+y_loc+h, x0:x0+w] = True
        else:
            cc = (labels[y_loc:y_loc+h, x0:x0+w] == i)
            chain_mask[y0+y_loc:y0+y_loc+h, x0:x0+w][cc] = 255

# Step 1b: dog tag detection (preserve in copy-back)
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
            print(f'[v319f] {band["name"]}: dog-tag CC area={a} w={w} h={h}')

# Step 2: TIGHT mask = letter CC bodies AND gray<80 (deep black cores only)
# Anti-alias edges (80-160) stay untouched. Chain + dog tag excluded.
text_mask = letter_mask & (gray < 80) & (chain_mask == 0) & (dog_tag_mask == 0)
print(f'[v319f] text mask px (gray<80 letter cores): {int(text_mask.sum())} (v319e was 140770)')

mask_vis = np.zeros((H, W, 3), dtype=np.uint8)
mask_vis[text_mask] = [255, 0, 0]
mask_vis[chain_mask > 0] = [0, 255, 0]
mask_vis[dog_tag_mask > 0] = [255, 255, 0]
overlay = (img_rgb * 0.5 + mask_vis * 0.5).astype(np.uint8)
Image.fromarray(overlay).save(str(Path(OUT_DIR) / 'mask_overlay.png'))
Image.fromarray(overlay[440:780, :, :]).save(str(Path(OUT_DIR) / 'mask_overlay_crop.png'))

mask_pil = Image.fromarray((text_mask.astype(np.uint8) * 255), mode='L')
src_pil = Image.fromarray(img_rgb)
print('[v319f] LaMa inpaint (mask tight, edge=1)...')
cleaned = lama_inpaint(src_pil, mask_pil, removal_strength=230, edge_smoothness=1)

# Step 3: defensive copy-back
cleaned_arr = np.array(cleaned).copy()
copy_back = (chain_mask > 0) | (dog_tag_mask > 0)
cleaned_arr[copy_back] = img_rgb[copy_back]
cleaned = Image.fromarray(cleaned_arr)
cleaned.save(str(Path(OUT_DIR) / 'armed_lama_cleaned.png'))

# Step 4: write new text
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
    desired_w = b['orig_w']
    spacing = max(0, (desired_w - nat_w) // max(len(text) - 1, 1))
    total_w = nat_w + spacing * (len(text) - 1)
    paste_x = b['orig_cx'] - total_w // 2 - bbox[0]
    paste_y = b['y1'] - bbox[3]
    BANDS.append({**b, 'fsize': fsize, 'font': font, 'bbox': bbox, 'nat_w': nat_w,
                  'spacing': spacing, 'total_w': total_w,
                  'paste_x': paste_x, 'paste_y': paste_y, 'capH': capH})
    print(f'[v319f] {b["name"]}: capH={capH} fsize={fsize} "{text}" nat_w={nat_w} spacing={spacing} '
          f'total_w={total_w} paste=({paste_x},{paste_y})')

for band in BANDS:
    draw.text((band['paste_x'], band['paste_y']),
              band['new_text'], font=band['font'], anchor='lt',
              fill=(20, 20, 20), spacing=band['spacing'])

OUT = Path(OUT_DIR) / 'armed_text_only_v319f.jpg'
out_pil.save(str(OUT), quality=95)

src_compare = Image.open(SRC).convert('RGB')
combined = Image.new('RGB', (src_compare.width * 2 + 30, src_compare.height), (50, 50, 50))
combined.paste(src_compare, (0, 0))
combined.paste(out_pil, (src_compare.width + 30, 0))
combined.save(str(Path(OUT_DIR) / 'armed_compare_v319f.jpg'), quality=90)

src_crop = src_compare.crop((0, 440, 1556, 780))
res_crop = out_pil.crop((0, 440, 1556, 780))
band_compare = Image.new('RGB', (src_crop.width * 2 + 20, src_crop.height), (60, 60, 60))
band_compare.paste(src_crop, (0, 0))
band_compare.paste(res_crop, (src_crop.width + 20, 0))
band_compare.save(str(Path(OUT_DIR) / 'armed_text_band_compare_v319f.jpg'), quality=95)
print(f'[v319f] Saved {OUT}')