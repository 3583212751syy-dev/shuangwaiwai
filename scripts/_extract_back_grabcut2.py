from PIL import Image, ImageFilter
import numpy as np
import cv2

src = "E:/Desktop/茶叶/产品背面图片.png"
out = "E:/Desktop/茶叶/成品_第九轮_真实站立包装/_assets/product_back_clean.png"

img = Image.open(src).convert("RGBA")
W, H = img.size
rgb = np.array(img.convert("RGB"))

# Create initial mask
mask = np.ones((H, W), dtype='uint8') * cv2.GC_PR_BGD  # probable background

# Outer border as definite background
border = 40
mask[:border, :] = cv2.GC_BGD
mask[-border:, :] = cv2.GC_BGD
mask[:, :border] = cv2.GC_BGD
mask[:, -border:] = cv2.GC_BGD

# Central rectangle as definite foreground (product area)
x1, x2 = int(W * 0.22), int(W * 0.78)
y1, y2 = int(H * 0.12), int(H * 0.88)
mask[y1:y2, x1:x2] = cv2.GC_FGD

# Inner probable foreground band
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

mask_img = Image.fromarray(final_mask, mode="L").filter(ImageFilter.GaussianBlur(0.5))

img.putalpha(mask_img)

# crop to content
bbox = img.getbbox()
if bbox:
    img = img.crop(bbox)

W2, H2 = img.size
pad = 20
new = Image.new("RGBA", (W2 + pad*2, H2 + pad*2), (255, 255, 255, 0))
new.paste(img, (pad, pad), img)

new.save(out, "PNG")
print("Saved clean product to", out)
