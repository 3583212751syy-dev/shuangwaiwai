"""Check v319c mask range in source image."""
import cv2
import numpy as np

def load_p(p):
    with open(p, 'rb') as f:
        d = f.read()
    return cv2.imdecode(np.frombuffer(d, dtype=np.uint8), cv2.IMREAD_COLOR)

src = load_p(r'E:/Desktop/双接口/image-fission/jobs/v315/source_armed.jpg')
gs = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
H, W = gs.shape

BANDS_DEF = [
    {'name': 'small', 'y0': 472, 'y1': 523},
    {'name': 'big1',  'y0': 541, 'y1': 640},
    {'name': 'big2',  'y0': 656, 'y1': 755},
]
chain_mask = np.zeros((H, W), dtype=np.uint8)
text_mask_global = np.zeros((H, W), dtype=bool)
for band in BANDS_DEF:
    y0, y1 = band['y0'], band['y1']
    band_h = y1 - y0
    band_gray = gs[y0:y1, :]
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

k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
text_mask_d = cv2.dilate(text_mask_global.astype(np.uint8) * 255, k, iterations=1) > 0
text_mask_d = text_mask_d & (chain_mask == 0)

print(f'mask total px: {int(text_mask_d.sum())}')
for band in BANDS_DEF:
    y0, y1 = band['y0'], band['y1']
    band_mask_px = int(text_mask_d[y0:y1, :].sum())
    band_total_px = (y1 - y0) * W
    print(f'  {band["name"]} y={y0}..{y1}: mask {band_mask_px} px ({band_mask_px/band_total_px*100:.1f}% of band)')

mask_vis = np.zeros((H, W, 3), dtype=np.uint8)
mask_vis[text_mask_d] = [255, 0, 0]
mask_vis[chain_mask > 0] = [0, 255, 0]
overlay = (src * 0.5 + mask_vis * 0.5).astype(np.uint8)
from PIL import Image
Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)).save(r'E:/Desktop/双接口/image-fission/jobs/v319c/mask_overlay.png')

# Crop just the text region for closer look
crop_src = src[440:780, :, :]
crop_mask = mask_vis[440:780, :, :]
crop_overlay = (crop_src * 0.5 + crop_mask * 0.5).astype(np.uint8)
Image.fromarray(cv2.cvtColor(crop_overlay, cv2.COLOR_RGB2BGR)).save(r'E:/Desktop/双接口/image-fission/jobs/v319c/mask_overlay_crop.png')