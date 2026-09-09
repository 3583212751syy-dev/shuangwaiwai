"""
v319v3: text-only fission for 3 source images — fixed sizing + per-image strategy

Lessons from v319v2:
  - Font scaling to FULL band width causes overflow (e.g., DUSKREVENANT fills 1552px)
  - Need natural font size + positioning, not horizontal stretch
  - MCKEART removal worked (verified by pixel check) — visual artifact was font overlap
  - Arc text "LA CASA DEL MURCIELAGO" and "Est. 1862" are decorative, not brand → skip
  - Eagle "JACKE DIANNIES" needs inpaint to preserve banner decoration (skull emblem)
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
    """Fill text region with local bg color (uniform bg only)."""
    out = arr_bgr.copy()
    out[y0:y1, x0:x1] = bg_color_bgr
    return out


def clean_band_inpaint(arr_bgr, y0, y1, x0, x1, text_mask_fn, keep_regions=()):
    """Inpaint text within band, excluding keep_regions (subjects/decorations)."""
    H, W = arr_bgr.shape[:2]
    band_mask = np.zeros((H, W), dtype=np.uint8)
    sub = arr_bgr[y0:y1, x0:x1]
    sub_mask = text_mask_fn(sub).astype(np.uint8) * 255
    band_mask[y0:y1, x0:x1] = sub_mask
    # Exclude keep regions
    for ky0, ky1 in keep_regions:
        keep_mask = np.zeros((H, W), dtype=np.uint8)
        keep_mask[ky0:ky1, :] = 255
        band_mask[keep_mask > 0] = 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    band_mask = cv2.dilate(band_mask, k, iterations=1)
    return cv2.inpaint(arr_bgr, band_mask, 5, cv2.INPAINT_NS)


def render_text_natural(word, font_path, fsize, color_bgr):
    """Render word at natural width (no horizontal stretch)."""
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    font = ImageFont.truetype(font_path, fsize)
    nat_w = int(round(font.getlength(word)))
    bbox_A = font.getbbox('A')
    cap_h = bbox_A[3] - bbox_A[1]
    cap_top = bbox_A[1]
    pad = abs(cap_top) + 10
    layer = Image.new('RGBA', (nat_w + 40, fsize + pad * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.text((20, pad - cap_top), word, font=font, fill=color_rgb + (255,))
    bbox = layer.getbbox()
    return layer.crop(bbox) if bbox else None


def paste_text(arr_bgr, layer_rgba, paste_x, paste_y):
    """Paste RGBA layer at (paste_x, paste_y) — cap top at paste_y."""
    H, W = arr_bgr.shape[:2]
    lw, lh = layer_rgba.size
    paste_x = max(0, min(paste_x, W - lw))
    paste_y = max(0, min(paste_y, H - lh))
    pil = Image.fromarray(cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)).convert('RGBA')
    pil.paste(layer_rgba, (paste_x, paste_y), layer_rgba)
    return cv2.cvtColor(np.array(pil.convert('RGB')), cv2.COLOR_RGB2BGR)


def make_compare(arr_src, arr_fission, out_path, labels=('ORIGINAL', 'FISSION v319v3')):
    src_rgb = cv2.cvtColor(arr_src, cv2.COLOR_BGR2RGB)
    final_rgb = cv2.cvtColor(arr_fission, cv2.COLOR_BGR2RGB)
    cmp_w = 800
    src_h = int(src_rgb.shape[0] * cmp_w / src_rgb.shape[1])
    final_h = int(final_rgb.shape[0] * cmp_w / final_rgb.shape[1])
    src_resized = cv2.resize(src_rgb, (cmp_w, src_h))
    final_resized = cv2.resize(final_rgb, (cmp_w, final_h))
    panel_h = 30
    compare = np.full((src_h + final_h + panel_h * 2 + 20, cmp_w, 3), 255, dtype=np.uint8)
    cv2.putText(compare, labels[0], (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
    compare[panel_h:panel_h + src_h, :] = src_resized
    cv2.putText(compare, labels[1], (10, panel_h + src_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 0), 2)
    compare[panel_h * 2 + src_h + 20:, :] = final_resized
    cv2_write(out_path, cv2.cvtColor(compare, cv2.COLOR_RGB2BGR))


# ============================================================
# Process each image
# ============================================================

def process_bacardi():
    print('\n=== BACARDI ===')
    src_path = r'E:\Desktop\双接口\image-fission\jobs\v319v_bat_style\source_bacardi.jpg'
    out_dir = r'E:\Desktop\双接口\image-fission\jobs\v319v_bat_style'
    bg = (182, 126, 171)  # BGR
    text = (30, 20, 40)  # BGR
    font_path = os.path.join(FONT_DIR, 'PlayfairDisplay-Bold.ttf')
    arr = cv2_read(src_path)
    H, W = arr.shape[:2]
    # Only remove brand text: BACARDI + MCKEART (skip arc + Est. 1862 — decorative)
    cleaned = arr.copy()
    # BACARDI: y=900-1000
    cleaned = clean_band_fill(cleaned, 900, 1000, 0, W, bg)
    # MCKEART: y=1000-1170
    cleaned = clean_band_fill(cleaned, 1000, 1170, 0, W, bg)
    # Add new text at NATURAL size, positioned at original locations
    # BACARDI original: y=900-1000 (h=100), so new text should have cap_h ~90
    # Playfair Display Bold: fsize=100 gives cap_h ~74
    # Render "NOCTAVEN" centered
    layer = render_text_natural('NOCTAVEN', font_path, 105, text)
    x = (W - layer.width) // 2
    cleaned = paste_text(cleaned, layer, x, 905)
    # MCKEART: y=1010-1160 (h=150), cap_h ~80
    layer = render_text_natural('DARK RESERVE', font_path, 90, text)
    x = (W - layer.width) // 2
    cleaned = paste_text(cleaned, layer, x, 1040)
    cv2_write(os.path.join(out_dir, 'fission.jpg'), cleaned)
    cv2_write(os.path.join(out_dir, 'cleaned.png'), arr)
    make_compare(arr, cleaned, os.path.join(out_dir, 'compare.jpg'))
    print(f'  done. (W={W},H={H})')


def process_eagle():
    print('\n=== EAGLE ===')
    src_path = r'E:\Desktop\双接口\image-fission\jobs\v319v_eagle\source.jpg'
    out_dir = r'E:\Desktop\双接口\image-fission\jobs\v319v_eagle'
    text = (235, 220, 200)  # BGR (light cream)
    font_path = os.path.join(FONT_DIR, 'BlackOpsOne-Regular.ttf')
    arr = cv2_read(src_path)
    H, W = arr.shape[:2]
    # Eagle text mask: light pixels (text is white/cream on dark bg)
    def text_mask(sub):
        g = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
        return (g > 200) & (g < 256)
    # Banner "JACKE DIANNIES": y=780-880, x=200-760 (full banner area)
    # Banner has skull+eagle emblem in middle that we want to preserve
    cleaned = arr.copy()
    # Banner: y=780-880, x=200-760 — but preserve skull emblem at center
    cleaned = clean_band_inpaint(cleaned, 780, 880, 200, 760, text_mask,
                                  keep_regions=((820, 870),))  # preserve center decoration
    # Bottom "TALIBAN LONDON DIT": y=970-1080
    cleaned = clean_band_inpaint(cleaned, 970, 1080, 100, 880, text_mask)
    # New text: "IRON EAGLES" on banner (centered, h~50)
    layer = render_text_natural('IRON EAGLES', font_path, 50, text)
    x = (W - layer.width) // 2
    cleaned = paste_text(cleaned, layer, x, 790)
    # "BORN TO RIDE" at bottom banner (h~50)
    layer = render_text_natural('BORN TO RIDE', font_path, 42, text)
    x = (W - layer.width) // 2
    cleaned = paste_text(cleaned, layer, x, 1030)
    cv2_write(os.path.join(out_dir, 'fission.jpg'), cleaned)
    cv2_write(os.path.join(out_dir, 'cleaned.png'), arr)
    make_compare(arr, cleaned, os.path.join(out_dir, 'compare.jpg'))
    print(f'  done. (W={W},H={H})')


def process_denim():
    print('\n=== DENIM ===')
    src_path = r'E:\Desktop\双接口\image-fission\jobs\v319v_denim\source.jpg'
    out_dir = r'E:\Desktop\双接口\image-fission\jobs\v319v_denim'
    bg = (245, 243, 240)  # BGR (off-white)
    text = (30, 40, 70)  # BGR (dark blue)
    font_path = os.path.join(FONT_DIR, 'ArchivoBlack-Regular.ttf')
    arr = cv2_read(src_path)
    H, W = arr.shape[:2]
    # UPGY: y=80-350, x=0-736
    cleaned = clean_band_fill(arr, 80, 350, 0, W, bg)
    # New text: "LUCKY" (4 letters matching UPGY length)
    # Original UPGY cap_h ~200, so use fsize that gives cap_h ~200
    # Archivo Black: fsize=190 → cap_h ~150
    layer = render_text_natural('LUCKY', font_path, 180, text)
    x = (W - layer.width) // 2
    cleaned = paste_text(cleaned, layer, x, 110)
    cv2_write(os.path.join(out_dir, 'fission.jpg'), cleaned)
    cv2_write(os.path.join(out_dir, 'cleaned.png'), arr)
    make_compare(arr, cleaned, os.path.join(out_dir, 'compare.jpg'))
    print(f'  done. (W={W},H={H})')


if __name__ == '__main__':
    process_bacardi()
    process_eagle()
    process_denim()
    print('\nAll done.')
