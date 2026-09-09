"""v319r: Per-band cropped LaMa inpaint to kill distance-transform haze.

v319q bug: distance_transform_edt fills letter regions with the NEAREST non-text
pixel. When the local camo around a letter is a LIGHT patch, the fill is a
light-camo "ghost" shaped like the old letter (visible blob). The fill is
faithful to the local camo but reads as a haze/ghosting.

v319r fix: for each text band, crop the band region (y0-margin..y1+margin,
full width) and run BIG-LA-MA inpaint on the cropped strip only. The strip
is ~1552x150 (area ~230k, comparable to the 800x256 strips that worked).
LaMa's Fourier texture synthesis fills the masked letter region with matching
camo, not a boundary pixel copy. This avoids:
  - Full-image LaMa failure (1552x2000 reconstructs everything as gray).
  - Distance-transform light-camo ghosts.

Chain (dog tag) is OUTSIDE the LaMa mask, so it stays pixel-perfect.
"""
import io
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps
from scipy.ndimage import distance_transform_edt

PROJECT = Path("E:/Desktop/双接口/image-fission")
SRC = PROJECT / "jobs" / "v315" / "source_armed.jpg"
OUT_DIR = PROJECT / "jobs" / "v319r"
ARIAL_BLACK = r"C:/Windows/Fonts/ariblk.ttf"
LAMA_MODEL = PROJECT / "ComfyUI" / "models" / "RMBG" / "Lama" / "big-lama.pt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Per-band: same as v319q (y, fsize, new text, etc.)
BANDS = [
    {'name': 'small', 'y0': 472, 'y1': 523, 'min_h': 40, 'min_a': 300, 'new_text': 'WE HONOR', 'orig_cx': 783, 'fsize': 71,  'spacing': 11,  'orig_w': 517},
    {'name': 'big1',  'y0': 541, 'y1': 640, 'min_h': 59, 'min_a': 1000, 'new_text': 'BRAVE',    'orig_cx': 829, 'fsize': 138, 'spacing': 198, 'orig_w': 1322},
    {'name': 'big2',  'y0': 656, 'y1': 755, 'min_h': 59, 'min_a': 1000, 'new_text': 'LEGION',   'orig_cx': 863, 'fsize': 138, 'spacing': 65,  'orig_w': 917},
]
MARGIN = 30  # px above/below each band included in crop for LaMa context


# ---------------------------------------------------------------------------
# LaMa loader + inpaint (proven v268 code, copied here for self-containment)
# ---------------------------------------------------------------------------
_LAMA = None


def load_lama():
    global _LAMA
    if _LAMA is not None:
        return _LAMA
    src = str(LAMA_MODEL)
    # torch.jit.load on Windows: C++ fopen rejects non-ASCII paths -> copy to ASCII tmp
    if any(ord(c) > 127 for c in src):
        tmp = os.path.join(tempfile.gettempdir(), "big-lama.pt")
        if not os.path.exists(tmp) or os.path.getsize(tmp) != os.path.getsize(src):
            shutil.copyfile(src, tmp)
        src = tmp
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(src, map_location=dev)
    model.eval()
    model.to(dev)
    _LAMA = (model, dev)
    print(f'[v319r] LaMa loaded on {dev}')
    return _LAMA


def _pad_to_8(img, is_mask=False):
    w, h = img.size
    nw = w + (8 - w % 8) if w % 8 else w
    nh = h + (8 - h % 8) if h % 8 else h
    if (nw, nh) == (w, h):
        return img
    fill = 0 if is_mask else None
    out = Image.new(img.mode, (nw, nh), color=fill)
    out.paste(img, (0, 0))
    return out


def lama_inpaint(src_pil, mask_pil, removal_strength=235, edge_smoothness=4):
    """src_pil: RGB. mask_pil: 'L' where white=inpaint. Returns cleaned RGB PIL
    at the same size as src_pil."""
    model, dev = load_lama()
    w, h = src_pil.size
    p_img = _pad_to_8(src_pil)
    p_mask = _pad_to_8(mask_pil, is_mask=True)
    if p_mask.size != p_img.size:
        p_mask = p_mask.resize(p_img.size, Image.LANCZOS)
    # v268 ComfyUI-node mask preprocessing: invert -> blur -> threshold
    p_mask = ImageOps.invert(p_mask)
    p_mask = p_mask.filter(ImageFilter.GaussianBlur(radius=edge_smoothness))
    p_mask = p_mask.point(lambda x: 0 if x > removal_strength else 255)
    img_t = torch.from_numpy(np.array(p_img).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)
    mask_t = torch.from_numpy(np.array(p_mask).astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(dev)
    with torch.inference_mode():
        res = model(img_t, mask_t)
    out = Image.fromarray((res[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
    if out.width > w or out.height > h:
        out = out.crop((0, 0, w, h))
    return out.convert("RGB")


# ---------------------------------------------------------------------------
# Mask building (same v319q shape-aware classification)
# ---------------------------------------------------------------------------
def build_masks(img_gray, H, W):
    letter_mask = np.zeros((H, W), dtype=bool)
    chain_mask = np.zeros((H, W), dtype=bool)
    for band in BANDS:
        y0, y1 = band['y0'], band['y1']
        band_h = y1 - y0
        band_gray = img_gray[y0:y1, :]
        dark = (band_gray < 80).astype(np.uint8) * 255
        n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
        for i in range(1, n):
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            a = stats[i, cv2.CC_STAT_AREA]
            x0 = stats[i, cv2.CC_STAT_LEFT]
            y_loc = stats[i, cv2.CC_STAT_TOP]
            aspect = w / max(h, 1)
            is_letter = (h >= band_h * 0.6) and (a >= 1000)
            is_thin_stroke = (h < band_h * 0.6) and ((aspect > 2.5) or (w > 15) or (a > 200))
            if is_letter or is_thin_stroke:
                letter_mask[y0+y_loc:y0+y_loc+h, x0:x0+w] = True
            else:
                cc = (labels[y_loc:y_loc+h, x0:x0+w] == i)
                chain_mask[y0+y_loc:y0+y_loc+h, x0:x0+w] |= cc
    return letter_mask, chain_mask


# ---------------------------------------------------------------------------
# Per-band LaMa inpaint
# ---------------------------------------------------------------------------
def perband_lama_clean(img_rgb, letter_mask, chain_mask):
    H, W = img_rgb.shape[:2]
    cleaned = img_rgb.copy()
    for band in BANDS:
        y0, y1 = band['y0'], band['y1']
        # Crop with margin (clamped to image)
        cy0 = max(0, y0 - MARGIN)
        cy1 = min(H, y1 + MARGIN)
        crop_rgb = img_rgb[cy0:cy1, :]
        # Letter mask for this band crop
        lmask_crop = letter_mask[cy0:cy1, :]
        cmask_crop = chain_mask[cy0:cy1, :]
        # Build PIL mask: 255 where letter, 0 where chain or background
        # Chain is EXCLUDED from inpaint (preserves dog tag).
        mask_arr = (lmask_crop & ~cmask_crop).astype(np.uint8) * 255
        mask_pil = Image.fromarray(mask_arr, mode='L')
        # Convert crop to PIL
        crop_pil = Image.fromarray(crop_rgb)
        # Skip LaMa if no letter pixels in crop
        if not lmask_crop.any():
            print(f'[v319r] {band["name"]}: no letter mask in crop, skip')
            continue
        ch, cw = crop_rgb.shape[:2]
        print(f'[v319r] {band["name"]}: LaMa inpaint crop {cw}x{ch} (mask px={int(lmask_crop.sum())})')
        cleaned_crop = lama_inpaint(crop_pil, mask_pil, removal_strength=235, edge_smoothness=4)
        # Paste back
        cleaned[cy0:cy1, :] = np.array(cleaned_crop)
    return cleaned


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_jpg(p):
    with open(p, 'rb') as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def main():
    print(f'[v319r] Loading source...')
    src_bgr = load_jpg(str(SRC))
    H, W = src_bgr.shape[:2]
    img_rgb = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)

    print(f'[v319r] Building masks...')
    letter_mask, chain_mask = build_masks(img_gray, H, W)
    print(f'[v319r] letter_mask px: {int(letter_mask.sum())}, chain_mask px: {int(chain_mask.sum())}')

    print(f'[v319r] Per-band LaMa inpaint...')
    cleaned_rgb = perband_lama_clean(img_rgb, letter_mask, chain_mask)
    Image.fromarray(cleaned_rgb).save(str(OUT_DIR / 'armed_lama_cleaned.png'))

    # Draw new text (Arial Black, one-pass per band, Y = y1 - ascent)
    out_pil = Image.fromarray(cleaned_rgb)
    draw = ImageDraw.Draw(out_pil)
    for band in BANDS:
        text = band['new_text']
        fsize = band['fsize']
        spacing = band['spacing']
        orig_cx = band['orig_cx']
        y1_band = band['y1']
        font = ImageFont.truetype(ARIAL_BLACK, fsize)
        ascent = font.getmetrics()[0]
        bbox = draw.textbbox((0, 0), text, font=font, spacing=spacing)
        total_w = bbox[2] - bbox[0]
        paste_x = orig_cx - total_w // 2 - bbox[0]
        paste_y = y1_band - ascent  # cap-bottom at band y1 (PIL default anchor "la")
        draw.text((paste_x, paste_y), text, font=font, fill=(20, 20, 20), spacing=spacing)
        print(f'[v319r] {band["name"]}: "{text}" paste=({paste_x},{paste_y}) total_w={total_w} ascent={ascent}')

    out_pil.save(str(OUT_DIR / 'armed_text_only_v319r.jpg'), quality=95)

    # Comparisons
    src_pil = Image.open(str(SRC)).convert('RGB')
    combined = Image.new('RGB', (src_pil.width * 2 + 30, src_pil.height), (50, 50, 50))
    combined.paste(src_pil, (0, 0))
    combined.paste(out_pil.convert('RGB'), (src_pil.width + 30, 0))
    combined.save(str(OUT_DIR / 'armed_compare_v319r.jpg'), quality=90)

    src_crop = src_pil.crop((0, 440, W, 780))
    res_crop = out_pil.crop((0, 440, W, 780))
    band_compare = Image.new('RGB', (src_crop.width * 2 + 20, src_crop.height), (60, 60, 60))
    band_compare.paste(src_crop, (0, 0))
    band_compare.paste(res_crop, (src_crop.width + 20, 0))
    band_compare.save(str(OUT_DIR / 'armed_text_band_compare_v319r.jpg'), quality=95)
    print(f'[v319r] Done.')


if __name__ == '__main__':
    main()
