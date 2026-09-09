"""
v319v2: text-only fission for 3 source images — CORRECT band positions + per-image strategy

User directive: '图片做好了就给我展示对比，没做好就跟我讲问题问我修改方式或者自己去git里找模型技能去处理'

Verified band positions (via diagnostic crops + density analysis):

BACARDI 1552x2000:
  arc        y=280-480   "LA CASA DEL MURCIELAGO" curve (dark text on pink bg)
  [SKIP]     y=540-880   BAT BADGE — DO NOT TOUCH (subject)
  main       y=890-1000  "BACARDI" (dark on pink, no bat overlap)
  sub        y=1000-1170 "MCKEART" (dark on pink)
  triangle   y=1180-1340 small triangle

EAGLE 964x1280:
  banner     y=700-870   "JACKE DIANNIES" on dark brown banner
  bottom     y=970-1080  "TALIBAN LONDON DIT" on dark banner
  [SKIP]     everywhere else = subject (eagle, skulls, flames, chains)

DENIM 736x1308:
  upgy       y=80-350    "UPGY" embroidered letters
  [SKIP]     y=400-1308  butterfly subject

Strategy per image:
  BACARDI: direct pink bg fill (uniform background)
  EAGLE:   cv2.inpaint within text region only (complex banner bg)
  DENIM:   direct white bg fill (uniform background)

Fonts:
  BACARDI: Playfair Display Bold (Didone serif matching original)
  EAGLE:   Black Ops One (military display matching gothic style)
  DENIM:   Archivo Black (heavy display matching denim patches)
"""
import os
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


def cv2_read(path):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def cv2_write(path, arr):
    cv2.imencode(os.path.splitext(path)[1] or '.png', arr)[1].tofile(path)


# ============================================================
# Image-specific configuration
# ============================================================

FONT_DIR = r'E:\Desktop\双接口\image-fission\fonts'

CONFIGS = {
    'bacardi': {
        'src': r'E:\Desktop\双接口\image-fission\jobs\v319v_bat_style\source_bacardi.jpg',
        'out_dir': r'E:\Desktop\双接口\image-fission\jobs\v319v_bat_style',
        'bg_color': (203, 159, 184),  # BGR for the pink background
        'text_color': (40, 30, 50),  # BGR for new text
        'text_mask_color': lambda arr: (cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) < 80),  # dark text
        'clean_method': 'fill',  # uniform bg
        'keep_regions': [(540, 880)],  # bat badge — don't touch
        'bands': [
            # (y0, y1, x0, x1, new_text, font, font_size, color_bgr)
            (280, 480, 0, 1552, 'DUSKREVENANT', os.path.join(FONT_DIR, 'PlayfairDisplay-Bold.ttf'), 80, (30, 20, 40)),
            (890, 1000, 0, 1552, 'NOCTAVEN', os.path.join(FONT_DIR, 'PlayfairDisplay-Bold.ttf'), 130, (30, 20, 40)),
            (1000, 1170, 0, 1552, 'DARK RESERVE', os.path.join(FONT_DIR, 'PlayfairDisplay-Bold.ttf'), 110, (30, 20, 40)),
        ],
    },
    'eagle': {
        'src': r'E:\Desktop\双接口\image-fission\jobs\v319v_eagle\source.jpg',
        'out_dir': r'E:\Desktop\双接口\image-fission\jobs\v319v_eagle',
        'bg_color': (25, 15, 10),  # BGR for dark banner bg
        'text_mask_color': lambda arr: (cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) > 200),  # light text
        'clean_method': 'inpaint',  # complex bg
        'keep_regions': [],  # no explicit keep; inpaint only inside text region
        'bands': [
            # (y0, y1, x0, x1, new_text, font, font_size, color_bgr)
            (700, 870, 200, 760, 'IRON EAGLES', os.path.join(FONT_DIR, 'BlackOpsOne-Regular.ttf'), 70, (240, 230, 220)),
            (970, 1080, 100, 880, 'BORN TO RIDE', os.path.join(FONT_DIR, 'BlackOpsOne-Regular.ttf'), 60, (240, 230, 220)),
        ],
    },
    'denim': {
        'src': r'E:\Desktop\双接口\image-fission\jobs\v319v_denim\source.jpg',
        'out_dir': r'E:\Desktop\双接口\image-fission\jobs\v319v_denim',
        'bg_color': (245, 243, 240),  # BGR for off-white bg
        'text_mask_color': lambda arr: (cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) < 180),  # dark text
        'clean_method': 'fill',
        'keep_regions': [(400, 1308)],  # butterfly
        'bands': [
            # (y0, y1, x0, x1, new_text, font, font_size, color_bgr)
            (80, 350, 0, 736, 'LUCKY', os.path.join(FONT_DIR, 'ArchivoBlack-Regular.ttf'), 220, (30, 40, 70)),
        ],
    },
}


# ============================================================
# Text cleaning
# ============================================================

def clean_band_fill(arr_bgr, y0, y1, x0, x1, bg_color_bgr):
    """Fill the text region with the local background color (uniform bg only)."""
    out = arr_bgr.copy()
    # Sample bg color from a few pixels at the edges of the band
    # (e.g., top-left of band if text is centered)
    samples = []
    for sy in [y0 + 5, y0 + 10, y1 - 5, y1 - 10]:
        for sx in [x0 + 5, x1 - 5]:
            if 0 <= sy < arr_bgr.shape[0] and 0 <= sx < arr_bgr.shape[1]:
                samples.append(arr_bgr[sy, sx])
    if samples:
        bg_color_bgr = tuple(np.median(samples, axis=0).astype(int).tolist())
    out[y0:y1, x0:x1] = bg_color_bgr
    return out


def clean_band_inpaint(arr_bgr, y0, y1, x0, x1, text_mask_fn, keep_regions=[]):
    """Inpaint only the text (detected by text_mask_fn) within band, excluding keep_regions."""
    H, W = arr_bgr.shape[:2]
    # Text mask within band
    band_mask = np.zeros((H, W), dtype=np.uint8)
    sub = arr_bgr[y0:y1, x0:x1]
    sub_mask = text_mask_fn(sub).astype(np.uint8) * 255
    band_mask[y0:y1, x0:x1] = sub_mask
    # Exclude keep regions (subjects)
    if keep_regions:
        keep_mask = np.zeros((H, W), dtype=np.uint8)
        for ky0, ky1 in keep_regions:
            keep_mask[ky0:ky1, :] = 255
        band_mask[keep_mask > 0] = 0
    # Dilate text mask slightly to capture anti-aliased edges
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    band_mask = cv2.dilate(band_mask, k, iterations=1)
    # Inpaint
    inpainted = cv2.inpaint(arr_bgr, band_mask, 5, cv2.INPAINT_NS)
    return inpainted, band_mask


# ============================================================
# Text rendering (PIL)
# ============================================================

def render_text(word, font_path, fsize, target_w, color_bgr, target_cap_h=None):
    """Render word in given font, scale to target_w, return RGBA layer with cap top at top edge."""
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    font = ImageFont.truetype(font_path, fsize)
    nat_w = int(round(font.getlength(word)))
    bbox_A = font.getbbox('A')
    cap_h = bbox_A[3] - bbox_A[1]
    cap_top = bbox_A[1]
    # Render with cap top near top
    pad = abs(cap_top) + 10
    layer = Image.new('RGBA', (nat_w + 40, fsize + pad * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.text((20, pad - cap_top), word, font=font, fill=color_rgb + (255,))
    bbox = layer.getbbox()
    if bbox:
        layer = layer.crop(bbox)
    # Scale horizontally to target_w
    if layer.width > 0 and target_w > 0:
        scale = target_w / layer.width
        if abs(scale - 1.0) > 0.01:
            new_w = int(round(layer.width * scale))
            new_h = int(round(layer.height * scale))
            layer = layer.resize((new_w, new_h), Image.LANCZOS)
    return layer


def paste_text(arr_bgr, layer_rgba, y0, x_center):
    """Paste the RGBA layer at (x_center, y0) — cap top at y0, x centered."""
    H, W = arr_bgr.shape[:2]
    lw, lh = layer_rgba.size
    x_paste = x_center - lw // 2
    # Composite the RGBA layer onto the BGR array
    pil = Image.fromarray(cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB))
    pil = pil.convert('RGBA')
    pil.paste(layer_rgba, (x_paste, y0), layer_rgba)
    return cv2.cvtColor(np.array(pil.convert('RGB')), cv2.COLOR_RGB2BGR)


# ============================================================
# Process each image
# ============================================================

def process_image(name, cfg):
    print(f'\n=== {name.upper()} ===')
    arr = cv2_read(cfg['src'])
    H, W = arr.shape[:2]
    print(f'  Source: {W}x{H}')
    # Step 1: Clean each band
    cleaned = arr.copy()
    for y0, y1, x0, x1, new_text, font_path, fsize, color_bgr in cfg['bands']:
        if cfg['clean_method'] == 'fill':
            cleaned = clean_band_fill(cleaned, y0, y1, x0, x1, cfg['bg_color'])
        else:  # inpaint
            cleaned, _ = clean_band_inpaint(cleaned, y0, y1, x0, x1, cfg['text_mask_color'], cfg.get('keep_regions', []))
    # Save cleaned intermediate
    cleaned_path = os.path.join(cfg['out_dir'], 'cleaned.png')
    cv2_write(cleaned_path, cleaned)
    print(f'  Cleaned: {cleaned_path}')
    # Step 2: Render new text
    final = cleaned.copy()
    for y0, y1, x0, x1, new_text, font_path, fsize, color_bgr in cfg['bands']:
        target_w = x1 - x0
        target_h = y1 - y0
        # Render text
        layer = render_text(new_text, font_path, fsize, target_w - 20, color_bgr, target_h)
        # Center x within the band
        x_center = (x0 + x1) // 2
        # Vertically center the cap within the band
        bbox_A = ImageFont.truetype(font_path, fsize).getbbox('A')
        cap_h = bbox_A[3] - bbox_A[1]
        cap_top = bbox_A[1]
        # Center cap within band
        paste_y = y0 + (target_h - cap_h) // 2 - cap_top
        # Clamp to within image
        paste_y = max(0, min(paste_y, H - layer.height))
        x_center = max(layer.width // 2, min(x_center, W - layer.width // 2))
        final = paste_text(final, layer, paste_y, x_center)
        print(f'  Band y={y0}..{y1}: "{new_text}" font_size={fsize} -> x={x_center} y={paste_y}')
    # Save final
    final_path = os.path.join(cfg['out_dir'], 'fission.jpg')
    cv2_write(final_path, final)
    print(f'  Final: {final_path}')
    # Save comparison side-by-side
    src_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    final_rgb = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)
    # Resize to common width
    cmp_w = 800
    src_h = int(src_rgb.shape[0] * cmp_w / src_rgb.shape[1])
    final_h = int(final_rgb.shape[0] * cmp_w / final_rgb.shape[1])
    src_resized = cv2.resize(src_rgb, (cmp_w, src_h))
    final_resized = cv2.resize(final_rgb, (cmp_w, final_h))
    panel_h = 30
    compare = np.full((src_h + final_h + panel_h * 2 + 20, cmp_w, 3), 255, dtype=np.uint8)
    # Label ORIGINAL
    cv2.putText(compare, 'ORIGINAL', (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
    compare[panel_h:panel_h + src_h, :] = src_resized
    # Label FISSION
    cv2.putText(compare, 'FISSION v319v2', (10, panel_h + src_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 0), 2)
    compare[panel_h * 2 + src_h + 20:panel_h * 2 + src_h + 20 + final_h, :] = final_resized
    compare_path = os.path.join(cfg['out_dir'], 'compare.jpg')
    cv2_write(compare_path, cv2.cvtColor(compare, cv2.COLOR_RGB2BGR))
    print(f'  Compare: {compare_path}')


if __name__ == '__main__':
    for name, cfg in CONFIGS.items():
        process_image(name, cfg)
    print('\nAll 3 images done.')
