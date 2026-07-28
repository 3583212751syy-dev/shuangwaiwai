from PIL import Image, ImageFilter
import numpy as np
import cv2

src = "E:/Desktop/茶叶/产品背面图片.png"
out = "E:/Desktop/茶叶/成品_第九轮_真实站立包装/_assets/product_back_clean.png"

img = Image.open(src).convert("RGBA")
W, H = img.size
rgb = np.array(img.convert("RGB"))

# Initial mask with grabCut
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
mask2, bgdModel, fgdModel = cv2.grabCut(rgb, mask, None, bgdModel, fgdModel, 8, cv2.GC_INIT_WITH_MASK)
final_mask = np.where((mask2 == cv2.GC_FGD) | (mask2 == cv2.GC_PR_FGD), 255, 0).astype('uint8')

# Morphological cleanup
kernel = np.ones((5,5), np.uint8)
final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel, iterations=1)

# Edge cleanup: make near-white edge pixels transparent
alpha = final_mask.copy().astype(float)
r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
maxc = np.maximum(np.maximum(r, g), b)
delta = maxc - np.minimum(np.minimum(r, g), b)
near_white = (maxc >= 235) & (delta <= 20)
# Only apply near-white removal at the edges of the mask
# Use distance transform to find edge pixels
dt = cv2.distanceTransform(final_mask, cv2.DIST_L2, 5)
edge = (dt < 8) & (final_mask > 0)
remove = edge & near_white
alpha[remove] = 0

# Erode mask slightly to remove residual edge halo
kernel2 = np.ones((3,3), np.uint8)
alpha = cv2.erode(alpha.astype(np.uint8), kernel2, iterations=1)

mask_img = Image.fromarray(alpha, mode="L").filter(ImageFilter.GaussianBlur(0.5))
img.putalpha(mask_img)

# Remove bottom shadow area: crop lower 5%
bbox = img.getbbox()
if bbox:
    left, top, right, bottom = bbox
    # remove a bit from bottom where shadow is
    bottom = int(bottom - (bottom - top) * 0.02)
    img = img.crop((left, top, right, bottom))

W2, H2 = img.size
pad = 20
new = Image.new("RGBA", (W2 + pad*2, H2 + pad*2), (255, 255, 255, 0))
new.paste(img, (pad, pad), img)

new.save(out, "PNG")
print("Saved clean product to", out)
