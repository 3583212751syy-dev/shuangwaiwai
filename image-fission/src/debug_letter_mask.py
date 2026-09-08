import cv2
import numpy as np
from pathlib import Path
from PIL import Image

def load_jpg(p):
    with open(p, 'rb') as f:
        d = f.read()
    return cv2.imdecode(np.frombuffer(d, dtype=np.uint8), cv2.IMREAD_COLOR)

def save_img(path, arr):
    """Use PIL to save (cv2.imwrite fails on unicode paths)."""
    if arr.ndim == 2:
        Image.fromarray(arr, mode='L').save(path)
    else:
        # cv2 is BGR, PIL is RGB
        if arr.shape[2] == 3:
            Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)).save(path)
        else:
            Image.fromarray(arr).save(path)

OUT = Path(r'E:\Desktop\双接口\image-fission\jobs\v319n')
OUT.mkdir(parents=True, exist_ok=True)

src = load_jpg(str(OUT.parent.parent / 'jobs' / 'v315' / 'source_armed.jpg'))
H, W = src.shape[:2]
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

BANDS = [{'y0': 472, 'y1': 523}, {'y0': 541, 'y1': 640}, {'y0': 656, 'y1': 755}]
letter_mask = np.zeros((H, W), dtype=bool)
for b in BANDS:
    y0, y1 = b['y0'], b['y1']
    band_h = y1 - y0
    band_gray = gray[y0:y1, :]
    dark = (band_gray < 80).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    for i in range(1, n):
        h = stats[i, cv2.CC_STAT_HEIGHT]
        a = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        x0 = stats[i, cv2.CC_STAT_LEFT]
        y_loc = stats[i, cv2.CC_STAT_TOP]
        if h >= band_h * 0.6 and a >= 1000:
            letter_mask[y0+y_loc:y0+y_loc+h, x0:x0+w] = True

mask_img = np.zeros((H, W, 3), dtype=np.uint8)
mask_img[letter_mask] = [255, 0, 0]
overlay = (cv2.cvtColor(src, cv2.COLOR_BGR2RGB) * 0.5 + mask_img * 0.5).astype(np.uint8)
save_img(str(OUT / 'letter_mask_overlay.png'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
save_img(str(OUT / 'letter_mask.png'), (letter_mask.astype(np.uint8) * 255))
print('saved')

