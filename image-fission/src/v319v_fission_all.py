"""v319v v3: Text-only fission — DIRECT FILL with background color (not LaMa).

LaMa fails on uniform backgrounds adjacent to complex objects (e.g. dark bat
body next to text): its Fourier synthesis pulls dark colors from the bat
into the surrounding masked area, creating dark blobs that look like text
residue. For uniform backgrounds, the correct approach is to sample the
background color and fill the text region directly.

For BACARDI: fill text rectangles with light-purple background, preserving
the bat body via ellipse exclusion. New text drawn on clean background.

For Eagle/Denim: fill text rectangles with local background. Subject is
separate from text, no exclusion needed.
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path("E:/Desktop/双接口/image-fission")
FONTDIR = PROJECT / "fonts"


def load_jpg(p):
    with open(p, 'rb') as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def build_rect_masks(H, W, bands, exclusion_ellipses=None):
    """Build full-image text mask: union of rectangular bands minus exclusion ellipses."""
    full = np.zeros((H, W), dtype=np.uint8)
    for band in bands:
        y0, y1 = band['y0'], band['y1']
        x0 = band.get('x0', 0)
        x1 = band.get('x1', W)
        full[y0:y1, x0:x1] = 255
    if exclusion_ellipses:
        Y, X = np.ogrid[:H, :W]
        for (cx, cy, rx, ry) in exclusion_ellipses:
            ellipse = ((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2 <= 1
            full[ellipse] = 0
    return full


def sample_bg_color(rgb, band, exclusion_ellipses=None, sample_pad=30):
    """Sample background color from a region just outside the text band.

    Uses the area above the band's top (or below the band's bottom), away
    from the band itself, to get a clean background color sample.
    For bands with exclusion ellipses (bat), samples from corners of the band
    that are far from the ellipse.
    """
    H, W = rgb.shape[:2]
    y0, y1 = band['y0'], band['y1']
    x0 = band.get('x0', 0)
    x1 = band.get('x1', W)
    # Sample from a strip just ABOVE the band (y0-sample_pad to y0)
    # and just BELOW (y1 to y1+sample_pad)
    samples = []
    sy0 = max(0, y0 - sample_pad)
    sy1 = max(0, y0)
    if sy1 > sy0:
        samples.append(rgb[sy0:sy1, x0:x1])
    sy0 = min(H, y1)
    sy1 = min(H, y1 + sample_pad)
    if sy1 > sy0:
        samples.append(rgb[sy0:sy1, x0:x1])
    if not samples:
        return np.array([128, 128, 128], dtype=np.uint8)
    all_pixels = np.concatenate([s.reshape(-1, 3) for s in samples], axis=0)
    # Median is robust to outliers
    return np.median(all_pixels, axis=0).astype(np.uint8)


def fill_text_regions(rgb, mask, bands, exclusion_ellipses=None):
    """Fill each text band with its local background color. Preserves excluded areas."""
    H, W = rgb.shape[:2]
    cleaned = rgb.copy()
    for band in bands:
        y0, y1 = band['y0'], band['y1']
        x0 = band.get('x0', 0)
        x1 = band.get('x1', W)
        # Get the mask for just this band
        band_mask = mask[y0:y1, x0:x1].copy()
        if band_mask.sum() < 50:
            print(f'  [{band["name"]}] mask too small, skip')
            continue
        # Sample background color
        bg_color = sample_bg_color(rgb, band, exclusion_ellipses)
        print(f'  [{band["name"]}] bg color RGB={tuple(bg_color.tolist())}, filling {int((band_mask>0).sum())} px')
        # Apply: set masked pixels to bg color
        region = cleaned[y0:y1, x0:x1]
        region[band_mask > 0] = bg_color
        cleaned[y0:y1, x0:x1] = region
    return cleaned


def render_text(img_pil, bands):
    out = img_pil.copy()
    draw = ImageDraw.Draw(out)
    for band in bands:
        text = band['new_text']
        font_path = band['font']
        fsize = band['fsize']
        orig_cx = band.get('orig_cx')
        if orig_cx is None:
            x0 = band.get('x0', 0)
            x1 = band.get('x1', img_pil.width)
            orig_cx = (x0 + x1) // 2
        y_top = band['y0']
        color = band.get('color', (20, 20, 20))
        try:
            font = ImageFont.truetype(font_path, fsize)
        except OSError:
            print(f'  [{band["name"]}] font not found, skip')
            continue
        bbox_A = font.getbbox("A")
        cap_top_offset = bbox_A[1]
        bbox_full = draw.textbbox((0, 0), text, font=font)
        tw = bbox_full[2] - bbox_full[0]
        paste_x = orig_cx - tw // 2 - bbox_full[0]
        paste_y = y_top - cap_top_offset
        spacing = band.get('spacing', 0)
        if spacing > 0:
            draw.text((paste_x, paste_y), text, font=font, fill=color, spacing=spacing)
        else:
            draw.text((paste_x, paste_y), text, font=font, fill=color)
        print(f'  [{band["name"]}] "{text}" fsize={fsize} -> w={tw} x={paste_x} y={paste_y}')
    return out


# ---- Image configs (same band positions as v2, but now using direct fill) ----

BACARDI_BANDS = [
    {'name': 'arc',      'y0': 330, 'y1': 470, 'x0': 250, 'x1': 1300, 'orig_cx': 776,
     'new_text': 'NOCTIS VENATOR', 'font': str(FONTDIR / 'PlayfairDisplay-Bold.ttf'),
     'fsize': 42, 'color': (30, 10, 50)},
    {'name': 'est',      'y0': 600, 'y1': 700, 'x0': 350, 'x1': 1200, 'orig_cx': 776,
     'new_text': 'EST. MMXXV', 'font': str(FONTDIR / 'PlayfairDisplay-Bold.ttf'),
     'fsize': 40, 'color': (30, 10, 50)},
    {'name': 'main',     'y0': 820, 'y1': 930, 'x0': 300, 'x1': 1250, 'orig_cx': 776,
     'new_text': 'NOCTAVEN', 'font': str(FONTDIR / 'PlayfairDisplay-Bold.ttf'),
     'fsize': 120, 'color': (20, 5, 40)},
    {'name': 'sub',      'y0': 1000, 'y1': 1160, 'x0': 400, 'x1': 1150, 'orig_cx': 776,
     'new_text': 'DARK RESERVE', 'font': str(FONTDIR / 'PlayfairDisplay-Bold.ttf'),
     'fsize': 72, 'color': (20, 5, 40)},
    {'name': 'triangle', 'y0': 1280, 'y1': 1410, 'x0': 650, 'x1': 900, 'orig_cx': 776,
     'new_text': 'AW', 'font': str(FONTDIR / 'PlayfairDisplay-Bold.ttf'),
     'fsize': 60, 'color': (20, 5, 40)},
]

BACARDI_EXCLUSIONS = [
    (776, 750, 270, 320),  # bat body
]

EAGLE_BANDS = [
    {'name': 'banner',   'y0': 680, 'y1': 800, 'x0': 180, 'x1': 780, 'orig_cx': 482,
     'new_text': 'IRON EAGLES', 'font': str(FONTDIR / 'BlackOpsOne-Regular.ttf'),
     'fsize': 64, 'color': (240, 230, 210)},
    {'name': 'bottom',   'y0': 1090, 'y1': 1220, 'x0': 120, 'x1': 850, 'orig_cx': 482,
     'new_text': 'BORN TO RIDE', 'font': str(FONTDIR / 'BlackOpsOne-Regular.ttf'),
     'fsize': 40, 'color': (220, 60, 40)},
]

DENIM_BANDS = [
    {'name': 'top',      'y0': 100, 'y1': 340, 'x0': 20, 'x1': 720, 'orig_cx': 368,
     'new_text': 'DENIM', 'font': str(FONTDIR / 'ArchivoBlack-Regular.ttf'),
     'fsize': 210, 'color': (30, 40, 80)},
]


def process_image(name, src_path, out_dir, bands, exclusions=None):
    print(f'\n=== {name}: {src_path} ===')
    src_bgr = load_jpg(str(src_path))
    H, W = src_bgr.shape[:2]
    rgb = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2RGB)
    print(f'  size: {W}x{H}')

    mask = build_rect_masks(H, W, bands, exclusions)
    print(f'  text mask px: {int((mask > 0).sum())}')
    Image.fromarray(mask).save(str(Path(out_dir) / 'debug_text_mask.png'))

    print(f'  filling text regions with background color...')
    cleaned_rgb = fill_text_regions(rgb, mask, bands, exclusions)
    Image.fromarray(cleaned_rgb).save(str(Path(out_dir) / 'cleaned.png'))

    print(f'  rendering text...')
    cleaned_pil = Image.fromarray(cleaned_rgb)
    final_pil = render_text(cleaned_pil, bands)
    final_pil.save(str(Path(out_dir) / 'fission.jpg'), quality=92)

    src_pil = Image.open(str(src_path)).convert('RGB')
    gap = 30
    comp = Image.new('RGB', (src_pil.width * 2 + gap, src_pil.height), (50, 50, 50))
    comp.paste(src_pil, (0, 0))
    comp.paste(final_pil.convert('RGB'), (src_pil.width + gap, 0))
    comp.save(str(Path(out_dir) / 'compare.jpg'), quality=88)
    print(f'  saved: {out_dir}/fission.jpg, compare.jpg')


def main():
    jobs = [
        ('BACARDI', PROJECT / 'jobs' / 'v319v_bat_style' / 'source_bacardi.jpg',
         PROJECT / 'jobs' / 'v319v_bat_style', BACARDI_BANDS, BACARDI_EXCLUSIONS),
        ('EAGLE',   PROJECT / 'jobs' / 'v319v_eagle' / 'source.jpg',
         PROJECT / 'jobs' / 'v319v_eagle',   EAGLE_BANDS,   None),
        ('DENIM',   PROJECT / 'jobs' / 'v319v_denim' / 'source.jpg',
         PROJECT / 'jobs' / 'v319v_denim',   DENIM_BANDS,   None),
    ]
    for name, src, out_dir, bands, exc in jobs:
        process_image(name, src, out_dir, bands, exc)
    print('\nAll done.')


if __name__ == '__main__':
    main()
