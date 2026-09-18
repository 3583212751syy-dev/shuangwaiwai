from PIL import Image, ImageFilter
import numpy as np
import cv2
from scipy import ndimage

src = "E:/Desktop/茶叶/产品背面图片.png"
out = "E:/Desktop/茶叶/成品_第九轮_真实站立包装/_assets/product_back_clean.png"

img = Image.open(src).convert("RGBA")
W, H = img.size
rgb = np.array(img.convert("RGB")).astype(np.float32)

# grabCut mask
mask = np.ones((H, W), dtype='uint8') * cv2.GC_PR_BGD
border = 40
mask[:border, :] = cv2.GC_BGD
mask[-border:, :] = cv2.GC_BGD
mask[:, :border] = cv2.GC_BGD
mask[:, -border:] = cv2.GC_BGD
x1, x2 = int(W * 0.22), int(W * 0.78)
y1, y2 = int(H * 0.12), int(H * 0.88)
mask[y1:y2, x1:x2] = cv2.GC_FGD
x1p, x2p = int(W * 0.15), int(W * 0.85)
y1p, y2p = int(H * 0.08), int(H * 0.92)
mask[y1p:y1, x1p:x2p] = cv2.GC_PR_FGD
mask[y2:y2p, x1p:x2p] = cv2.GC_PR_FGD
mask[y1p:y2p, x1p:x1] = cv2.GC_PR_FGD
mask[y1p:y2p, x2:x2p] = cv2.GC_PR_FGD

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
mask2, _, _ = cv2.grabCut(rgb.astype(np.uint8), mask, None, bgdModel, fgdModel, 8, cv2.GC_INIT_WITH_MASK)
hard_mask = ((mask2 == cv2.GC_FGD) | (mask2 == cv2.GC_PR_FGD)).astype(np.uint8)

# Keep only largest connected component
labeled, num = ndimage.label(hard_mask)
if num > 0:
    sizes = ndimage.sum(hard_mask, labeled, range(1, num+1))
    largest = np.argmax(sizes) + 1
    hard_mask = (labeled == largest).astype(np.uint8)

# Distance to white for soft edges
dist_to_white = np.sqrt(np.sum((255.0 - rgb) ** 2, axis=2))
alpha = np.clip(dist_to_white * 3.0, 0, 255).astype(np.uint8)
alpha[hard_mask == 0] = 0

# Erode slightly
kernel = np.ones((3,3), np.uint8)
alpha = cv2.erode(alpha, kernel, iterations=1)

mask_img = Image.fromarray(alpha, mode="L").filter(ImageFilter.GaussianBlur(0.5))
img.putalpha(mask_img)

# crop
bbox = img.getbbox()
if bbox:
    left, top, right, bottom = bbox
    bottom = int(bottom - (bottom - top) * 0.015)
    img = img.crop((left, top, right, bottom))

W2, H2 = img.size
pad = 20
new = Image.new("RGBA", (W2 + pad*2, H2 + pad*2), (255, 255, 255, 0))
new.paste(img, (pad, pad), img)

new.save(out, "PNG")
print("Saved clean product to", out)
