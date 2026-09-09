"""
v319v4: text-only fission for 3 source images — CORRECT band positions + bigger fonts

Verified Y positions from diagnostic crops:

BACARDI 1552x2000:
  [KEEP]   y=280-880  arc + bat badge + Est. 1862 (decorative, not brand)
  main     y=1000-1180  "BACARDI" (NOT y=900-1000 as I had wrong before)
  sub      y=1180-1340  "MCKEART"
  tri      y=1380-1500  small triangle (could be removed)

EAGLE 964x1280:
  banner   y=780-880   "JACKE DIANNIES" on dark banner (preserve skull emblem)
  bottom   y=970-1080  "TALIBAN LONDON DIT" on dark banner

DENIM 736x1308:
  upgy     y=80-350    "UPGY" embroidered letters
"""
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


def cv2_read(path):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def cv2_write(path, arr):
    cv2.imencode(os.path.splitext(path)[1] or '.png', arr)[1].tofile(path)


FONT_DIR = r'E:\Desktop\双接口\image-fission\fonts'


def clean_band_fill(arr_bgr, y0, y1, x0, x1, bg_color_bgr):
    out = arr_bgr.copy()
    out[y0:y1, x0:x1] = bg_color_bgr
    return out


def clean_band_inpaint(arr_bgr, y0, y1, x0, x1, text_mask_fn, keep_regions=()):
    H, W = arr_bgr.shape[:2]
    band_mask = np.zeros((H, W), dtype=np.uint8)
    sub = arr_bgr[y0:y1, x0:x1]
    sub_mask = text_mask_fn(sub).astype(np.uint8) * 255
    band_mask[y0:y1, x0:x1] = sub_mask
    for ky0, ky1 in keep_regions:
        keep_mask = np.zeros((H, W), dtype=np.uint8)
        keep_mask[ky0:ky1, :] = 255
        band_mask[keep_mask > 0] = 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    band_mask = cv2.dilate(band_mask, k, iterations=1)
    return cv2.inpaint(arr_bgr, band_mask, 5, cv2.INPAINT_NS)


def render_text_natural(word, font_path, fsize, color_bgr):
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    font = ImageFont.truetype(font_path, fsize)
    bbox_A = font.getbbox('A')
    pad = abs(bbox_A[1]) + 10
    layer = Image.new('RGBA', (int(font.getlength(word)) + 40, fsize + pad * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.text((20, pad - bbox_A[1]), word, font=font, fill=color_rgb + (255,))
    bbox = layer.getbbox()
    return layer.crop(bbox) if bbox else None


def paste_text(arr_bgr, layer_rgba, paste_x, paste_y):
    H, W = arr_bgr.shape[:2]
    lw, lh = layer_rgba.size
    paste_x = max(0, min(paste_x, W - lw))
    paste_y = max(0, min(paste_y, H - lh))
    pil = Image.fromarray(cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)).convert('RGBA')
    pil.paste(layer_rgba, (paste_x, paste_y), layer_rgba)
    return cv2.cvtColor(np.array(pil.convert('RGB')), cv2.COLOR_RGB2BGR)


def make_compare(arr_src, arr_fission, out_path, label='FISSION v319v4'):
    src_rgb = cv2.cvtColor(arr_src, cv2.COLOR_BGR2RGB)
    final_rgb = cv2.cvtColor(arr_fission, cv2.COLOR_BGR2RGB)
    cmp_w = 600
    src_h = int(src_rgb.shape[0] * cmp_w / src_rgb.shape[1])
    final_h = int(final_rgb.shape[0] * cmp_w / final_rgb.shape[1])
    src_r = cv2.resize(src_rgb, (cmp_w, src_h))
    final_r = cv2.resize(final_rgb, (cmp_w, final_h))
    panel_h = 30
    H = max(src_h, final_h) + panel_h
    compare = np.full((H * 2 + 20, cmp_w * 2 + 20, 3), 255, dtype=np.uint8)
    cv2.putText(compare, 'ORIGINAL', (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
    compare[panel_h:panel_h + src_h, 0:cmp_w] = src_r
    cv2.putText(compare, label, (cmp_w + 30, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 0), 2)
    compare[panel_h:panel_h + final_h, cmp_w + 20:] = final_r
    cv2_write(out_path, cv2.cvtColor(compare, cv2.COLOR_RGB2BGR))


# ============================================================
# Process each image
# ============================================================

def process_bacardi():
    print('\n=== BACARDI ===')
    src_path = r'E:\Desktop\双接口\image-fission\jobs\v319v_bat_style\source_bacardi.jpg'
    out_dir = r'E:\Desktop\双接口\image-fission\jobs\v319v_bat_style'
    bg = (182, 126, 171)  # BGR (pink)
    text = (30, 20, 40)  # BGR (near-black)
    font_path = os.path.join(FONT_DIR, 'PlayfairDisplay-Bold.ttf')
    arr = cv2_read(src_path)
    H, W = arr.shape[:2]
    print(f'  Source: {W}x{H}')
    cleaned = arr.copy()
    # CORRECT bands: BACARDI at y=1000-1180, MCKEART at y=1180-1340
    cleaned = clean_band_fill(cleaned, 1000, 1180, 0, W, bg)
    cleaned = clean_band_fill(cleaned, 1180, 1340, 0, W, bg)
    # New text: NOCTAVEN (replaces BACARDI), DARK RESERVE (replaces MCKEART)
    # BACARDI cap_h ~140 in 180px band → Playfair Bold fsize=180 → cap_h ~135
    # MCKEART cap_h ~120 in 160px band → fsize=155 → cap_h ~115
    layer = render_text_natural('NOCTAVEN', font_path, 180, text)
    x = (W - layer.width) // 2
    cleaned = paste_text(cleaned, layer, x, 1010)
    print(f'  NOCTAVEN: x={x} y=1010 w={layer.width} h={layer.height}')
    layer = render_text_natural('DARK RESERVE', font_path, 150, text)
    x = (W - layer.width) // 2
    cleaned = paste_text(cleaned, layer, x, 1195)
    print(f'  DARK RESERVE: x={x} y=1195 w={layer.width} h={layer.height}')
    cv2_write(os.path.join(out_dir, 'fission.jpg'), cleaned)
    make_compare(arr, cleaned, os.path.join(out_dir, 'compare.jpg'))


def process_eagle():
    print('\n=== EAGLE ===')
    src_path = r'E:\Desktop\双接口\image-fission\jobs\v319v_eagle\source.jpg'
    out_dir = r'E:\Desktop\双接口\image-fission\jobs\v319v_eagle'
    text = (235, 220, 200)  # BGR
    font_path = os.path.join(FONT_DIR, 'BlackOpsOne-Regular.ttf')
    arr = cv2_read(src_path)
    H, W = arr.shape[:2]
    print(f'  Source: {W}x{H}')
    def text_mask(sub):
        g = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
        return (g > 180) & (g < 256)
    cleaned = arr.copy()
    # Banner: y=780-880, x=200-760, preserve center decoration (820-870 in y, but banner is y=780-880)
    # Need to be careful — the skull-eagle emblem is INSIDE the banner area
    cleaned = clean_band_inpaint(cleaned, 780, 880, 200, 760, text_mask, keep_regions=((815, 875),))
    # Bottom: y=970-1080
    cleaned = clean_band_inpaint(cleaned, 970, 1080, 100, 880, text_mask)
    layer = render_text_natural('IRON EAGLES', font_path, 55, text)
    x = (W - layer.width) // 2
    cleaned = paste_text(cleaned, layer, x, 790)
    print(f'  IRON EAGLES: x={x} y=790 w={layer.width} h={layer.height}')
    layer = render_text_natural('BORN TO RIDE', font_path, 45, text)
    x = (W - layer.width) // 2
    cleaned = paste_text(cleaned, layer, x, 1010)
    print(f'  BORN TO RIDE: x={x} y=1010 w={layer.width} h={layer.height}')
    cv2_write(os.path.join(out_dir, 'fission.jpg'), cleaned)
    make_compare(arr, cleaned, os.path.join(out_dir, 'compare.jpg'))


def process_denim():
    print('\n=== DENIM ===')
    src_path = r'E:\Desktop\双接口\image-fission\jobs\v319v_denim\source.jpg'
    out_dir = r'E:\Desktop\双接口\image-fission\jobs\v319v_denim'
    bg = (245, 243, 240)
    text = (140, 110, 60)  # BGR denim-blue (matching UPGY's actual color)
    font_path = os.path.join(FONT_DIR, 'ArchivoBlack-Regular.ttf')
    arr = cv2_read(src_path)
    H, W = arr.shape[:2]
    print(f'  Source: {W}x{H}')
    cleaned = clean_band_fill(arr, 80, 320, 0, W, bg)
    # UPGY cap_h ~200 in 240px band → Archivo Black fsize=150 → cap_h ~115
    layer = render_text_natural('LUCKY', font_path, 150, text)
    x = (W - layer.width) // 2
    cleaned = paste_text(cleaned, layer, x, 100)
    print(f'  LUCKY: x={x} y=100 w={layer.width} h={layer.height}')
    cv2_write(os.path.join(out_dir, 'fission.jpg'), cleaned)
    make_compare(arr, cleaned, os.path.join(out_dir, 'compare.jpg'))


if __name__ == '__main__':
    process_bacardi()
    process_eagle()
    process_denim()
    print('\nAll done.')
