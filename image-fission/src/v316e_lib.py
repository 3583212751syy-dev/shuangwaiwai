"""
Shared module extracted from v316e — text detection + mask generation
for ARMED source image.

Usage:
    import v316e_lib as lib
    img_rgb, fill_mask, BANDS, H, W = lib.prepare_text_mask(SRC)
"""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

def load_jpg(p):
    with open(p, 'rb') as f:
        data = f.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

# ARMED source text bands (measured from source_armed.jpg 1556x2000)
BANDS = [
    {'name': 'small', 'y0': 472, 'y1': 523, 'new_text': 'HEROES'},
    {'name': 'big1',  'y0': 541, 'y1': 640, 'new_text': 'BRAVE'},
    {'name': 'big2',  'y0': 656, 'y1': 755, 'new_text': 'LEGION'},
]

def prepare_text_mask(src_path, with_inpaint_fill=False):
    """Load source, detect text vs chain, return (img_rgb, fill_mask).

    fill_mask: bool (H, W), True where text pixels are (to be filled by
    inpaint or nearest-camo).
    """
    img_bgr = load_jpg(src_path)
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

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

    # Dilate text mask (catches anti-aliased edges), subtract chain
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    text_mask_d = cv2.dilate(text_mask_global.astype(np.uint8) * 255, k, iterations=1) > 0
    text_mask_d = text_mask_d & (chain_mask == 0)
    return img_rgb, text_mask_d, H, W