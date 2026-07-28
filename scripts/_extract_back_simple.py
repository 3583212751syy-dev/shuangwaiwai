from PIL import Image, ImageFilter
import numpy as np
from scipy import ndimage

src = "E:/Desktop/茶叶/产品背面图片.png"
out = "E:/Desktop/茶叶/成品_第九轮_真实站立包装/_assets/product_back_clean.png"

img = Image.open(src).convert("RGBA")
W, H = img.size
rgb = np.array(img.convert("RGB")).astype(np.float32)

# Distance to pure white
dist = np.sqrt(np.sum((255.0 - rgb) ** 2, axis=2))

# Initial mask: pixels sufficiently far from white
mask = (dist > 30).astype(np.uint8)

# Keep largest connected component
labeled, num = ndimage.label(mask)
if num > 0:
    sizes = ndimage.sum(mask, labeled, range(1, num+1))
    largest = np.argmax(sizes) + 1
    mask = (labeled == largest).astype(np.uint8)

# Fill holes
mask = ndimage.binary_fill_holes(mask).astype(np.uint8)

# Soft alpha from distance
alpha = np.clip((dist - 15) * 3.0, 0, 255)
alpha[mask == 0] = 0

# Slight erode to clean edge
import cv2
kernel = np.ones((3,3), np.uint8)
alpha = cv2.erode(alpha.astype(np.uint8), kernel, iterations=1)

mask_img = Image.fromarray(alpha.astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(0.5))
img.putalpha(mask_img)

# crop
bbox = img.getbbox()
if bbox:
    left, top, right, bottom = bbox
    bottom = int(bottom - (bottom - top) * 0.01)
    img = img.crop((left, top, right, bottom))

W2, H2 = img.size
pad = 20
new = Image.new("RGBA", (W2 + pad*2, H2 + pad*2), (255, 255, 255, 0))
new.paste(img, (pad, pad), img)

new.save(out, "PNG")
print("Saved clean product to", out)
