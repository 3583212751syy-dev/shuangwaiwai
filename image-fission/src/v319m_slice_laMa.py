"""v319m: Slice ARMED into 3 horizontal strips (one per text band + context),
run LaMa on each strip separately (proven working at 800x256), then recombine.

Why this works:
  - v319c/v319l/v319k all failed: LaMa on 1552x2000 full ARMED rebuilt the whole
    image because Fourier conv sees 100% high-freq periodic texture with no
    "clean" reference.
  - v319l test_crop (800x256) WORKED: LaMa only sees the text band + small camo
    context, so Fourier conv can sample from clean camo in that region.

Pipeline:
  1. Slice into 3 strips: y=440-770 (each text band + ~120px context above/below)
  2. Build tight mask for letters in that strip
  3. Run LaMa per strip
  4. Recombine strips back into 1552x2000
  5. Write new text at original Y positions

Output: jobs/v319m/armed_text_only_v319m.jpg
"""
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter

PROJECT = Path("E:/Desktop/双接口/image-fission")
SRC = PROJECT / "jobs" / "v315" / "source_armed.jpg"
OUT_DIR = PROJECT / "jobs" / "v319m"
LAMA_PT = PROJECT / "ComfyUI" / "models" / "RMBG" / "Lama" / "big-lama.pt"
ARIAL_BLACK = r"C:/Windows/Fonts/ariblk.ttf"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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
    return model


def pad8(img):
    if isinstance(img, np.ndarray):
        h, w = img.shape[:2]
    else:
        w, h = img.size
    nw, nh = ((w + 7) // 8) * 8, ((h + 7) // 8) * 8
    if isinstance(img, np.ndarray):
        canvas = np.zeros((nh, nw, img.shape[2]) if img.ndim == 3 else (nh, nw), dtype=img.dtype)
        canvas[:h, :w] = img
        return canvas
    else:
        canvas = Image.new(img.mode, (nw, nh), 0)
        canvas.paste(img, (0, 0))
        return canvas


def lama_inpaint_strip(model, img_rgb_strip, mask_strip):
    """Run LaMa on a small strip. Returns cleaned RGB uint8 of same size as input."""
    H, W = img_rgb_strip.shape[:2]
    # Use v268 mask pipeline (invert + blur + threshold) to make soft edge
    mask_pil = Image.fromarray(mask_strip.astype(np.uint8) * 255, mode='L')
    mask_inv = ImageOps.invert(mask_pil)
    mask_blur = mask_inv.filter(ImageFilter.GaussianBlur(radius=4))
    mask_thresh = mask_blur.point(lambda x: 0 if x > 230 else 255)
    mask_final = np.array(mask_thresh, dtype=np.uint8)

    img_pil = Image.fromarray(img_rgb_strip)
    p_img = pad8(img_pil)
    p_mask = pad8(Image.fromarray(mask_final))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    img_t = torch.from_numpy(np.array(p_img).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
    mask_t = torch.from_numpy(np.array(p_mask).astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(dev)
    with torch.inference_mode():
        res = model(img_t, mask_t)
    out = (res[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return out[:H, :W]


def build_letter_mask(gray_strip, y_offset_global, band_def):
    """Build tight letter mask within a strip. y_offset_global = strip's y in original image."""
    y0_local = max(0, band_def['y0'] - y_offset_global)
    y1_local = min(gray_strip.shape[0], band_def['y1'] - y_offset_global)
    band_h = band_def['y1'] - band_def['y0']
    band_gray = gray_strip[y0_local:y1_local, :]
    dark = (band_gray < 80).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    letter_mask = np.zeros(gray_strip.shape, dtype=np.uint8)
    for i in range(1, n):
        h = stats[i, cv2.CC_STAT_HEIGHT]
        a = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        # Strict: only full-height letters
        if h >= band_h * 0.6 and a >= 1000:
            x0 = stats[i, cv2.CC_STAT_LEFT]
            y_loc = stats[i, cv2.CC_STAT_TOP]
            letter_mask[y0_local + y_loc:y0_local + y_loc + h, x0:x0 + w] = 255
    return letter_mask


print(f'[v319m] Loading source...')
src_bgr = load_jpg(str(SRC))
H, W = src_bgr.shape[:2]
print(f'  source: {W}x{H}')
img_rgb = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)

# Strip definition: y range of each strip in the original image
# Width is split into 2 halves (x=0..800 and x=720..1556) to keep strips ~800px wide
# (LaMa Fourier conv over 1556px wide sees too much camo context and rebuilds the
# entire image; 800px strips work as proven by v319l test_crop).
STRIPS = [
    # small band y=472-523
    {'name': 'small_L', 'y0': 400, 'y1': 580, 'x0': 0,    'x1': 800},
    {'name': 'small_R', 'y0': 400, 'y1': 580, 'x0': 720,  'x1': 1556},
    # big1 band y=541-640
    {'name': 'big1_L',  'y0': 480, 'y1': 700, 'x0': 0,    'x1': 800},
    {'name': 'big1_R',  'y0': 480, 'y1': 700, 'x0': 720,  'x1': 1556},
    # big2 band y=656-755
    {'name': 'big2_L',  'y0': 600, 'y1': 820, 'x0': 0,    'x1': 800},
    {'name': 'big2_R',  'y0': 600, 'y1': 820, 'x0': 720,  'x1': 1556},
]

BANDS = [
    {'name': 'small', 'y0': 472, 'y1': 523, 'new_text': 'WE HONOR', 'orig_cx': 783, 'fsize': 71, 'spacing': 11, 'orig_w': 517},
    {'name': 'big1',  'y0': 541, 'y1': 640, 'new_text': 'BRAVE',    'orig_cx': 829, 'fsize': 138, 'spacing': 198, 'orig_w': 1322},
    {'name': 'big2',  'y0': 656, 'y1': 755, 'new_text': 'LEGION',   'orig_cx': 863, 'fsize': 138, 'spacing': 65, 'orig_w': 917},
]

# Map band -> strips (each band processed in 2 halves: _L and _R)
BAND_TO_STRIPS = {
    'small': ['small_L', 'small_R'],
    'big1':  ['big1_L',  'big1_R'],
    'big2':  ['big2_L',  'big2_R'],
}

print(f'[v319m] Loading LaMa model...')
model = load_lama(LAMA_PT)

# For each strip, build its mask and run LaMa
strip_results = {}
for strip in STRIPS:
    s_name = strip['name']
    s_y0, s_y1 = strip['y0'], strip['y1']
    s_x0, s_x1 = strip['x0'], strip['x1']
    strip_img = img_rgb[s_y0:s_y1, s_x0:s_x1].copy()
    strip_gray = gray[s_y0:s_y1, s_x0:s_x1]

    # Build mask: union of letter masks for all bands in this strip
    strip_mask = np.zeros(strip_gray.shape, dtype=np.uint8)
    bands_in_strip = [b for b in BANDS if s_name in BAND_TO_STRIPS[b['name']]]
    for band in bands_in_strip:
        letter_m = build_letter_mask(strip_gray, s_y0, band)
        strip_mask = np.maximum(strip_mask, letter_m)

    print(f'[v319m] Strip {s_name} y={s_y0}-{s_y1} x={s_x0}-{s_x1} '
          f'({s_y1-s_y0}x{s_x1-s_x0}): mask px={int((strip_mask>0).sum())}, '
          f'bands={[b["name"] for b in bands_in_strip]}')

    # Save mask for inspection
    mask_vis = (strip_mask > 0).astype(np.uint8) * 255
    Image.fromarray(mask_vis, mode='L').save(str(OUT_DIR / f'mask_{s_name}.png'))

    # Run LaMa
    cleaned = lama_inpaint_strip(model, strip_img, strip_mask)
    strip_results[s_name] = (s_y0, s_y1, s_x0, s_x1, cleaned)
    Image.fromarray(cleaned).save(str(OUT_DIR / f'cleaned_{s_name}.png'))

# Recombine: start with source, replace each strip region with cleaned version
final_rgb = img_rgb.copy()
for s_name, (s_y0, s_y1, s_x0, s_x1, cleaned) in strip_results.items():
    final_rgb[s_y0:s_y1, s_x0:s_x1] = cleaned

# Save the LaMa-only cleaned base (no new text yet)
Image.fromarray(final_rgb).save(str(OUT_DIR / 'armed_lama_cleaned.png'))
print(f'[v319m] Saved LaMa cleaned base')

# Now write new text on the cleaned base
out_pil = Image.fromarray(final_rgb)
draw = ImageDraw.Draw(out_pil)

for band in BANDS:
    text = band['new_text']
    fsize = band['fsize']
    spacing = band['spacing']
    orig_cx = band['orig_cx']
    y0_band = band['y0']
    y1_band = band['y1']
    font = ImageFont.truetype(ARIAL_BLACK, fsize)
    # Bottom-align to band baseline
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    # PIL spacing adds N px between each character
    # We need to compute total width with spacing
    n_chars = len(text)
    total_w = text_w + spacing * (n_chars - 1) if spacing > 0 else text_w
    paste_x = orig_cx - total_w // 2 - bbox[0]
    paste_y = y1_band - text_h - bbox[1]
    # Render with spacing using overlay technique
    txt_layer = Image.new('RGBA', (total_w + 20, text_h + 20), (0, 0, 0, 0))
    txt_draw = ImageDraw.Draw(txt_layer)
    cur_x = 10 - bbox[0]
    for ch in text:
        txt_draw.text((cur_x, 10 - bbox[1]), ch, font=font, fill=(20, 20, 20, 255))
        ch_bbox = font.getbbox(ch)
        cur_x += (ch_bbox[2] - ch_bbox[0]) + spacing
    out_pil.paste(txt_layer, (paste_x, paste_y), txt_layer)
    print(f'[v319m] Wrote {band["name"]} "{text}" at ({paste_x},{paste_y}) w={total_w}')

OUT = OUT_DIR / 'armed_text_only_v319m.jpg'
out_pil.save(str(OUT), quality=95)
print(f'[v319m] Saved {OUT}')

# Compare
src_pil = Image.open(str(SRC)).convert('RGB')
combined = Image.new('RGB', (src_pil.width * 2 + 30, src_pil.height), (50, 50, 50))
combined.paste(src_pil, (0, 0))
combined.paste(out_pil.convert('RGB'), (src_pil.width + 30, 0))
combined.save(str(OUT_DIR / 'armed_compare_v319m.jpg'), quality=90)

# Text band closeup
src_crop = src_pil.crop((0, 440, W, 780))
res_crop = out_pil.crop((0, 440, W, 780))
band_compare = Image.new('RGB', (src_crop.width * 2 + 20, src_crop.height), (60, 60, 60))
band_compare.paste(src_crop, (0, 0))
band_compare.paste(res_crop, (src_crop.width + 20, 0))
band_compare.save(str(OUT_DIR / 'armed_text_band_compare_v319m.jpg'), quality=95)
print(f'[v319m] Saved compare images')
