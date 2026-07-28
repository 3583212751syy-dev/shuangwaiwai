from PIL import Image, ImageFilter
import numpy as np
import cv2
from scipy import ndimage

src = "E:/Desktop/茶叶/产品背面图片.png"
out = "E:/Desktop/茶叶/成品_第九轮_真实站立包装/_assets/product_back_clean.png"

img = Image.open(src).convert("RGBA")
W, H = img.size
rgb = np.array(img.convert("RGB")).astype(np.float32)

# Convert to HSV for saturation check
hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
sat = hsv[:, :, 1].astype(np.float32)
val = hsv[:, :, 2].astype(np.float32)

# Distance to pure white
dist_white = np.sqrt(np.sum((255.0 - rgb) ** 2, axis=2))

# Foreground: far from white AND has some color (saturation) OR very dark (text/barcode)
# Background/shadow: near white or gray with low saturation
# Tighter distance to exclude gray shadows; keep dark text/barcode via val check
mask = (
    (dist_white > 45) &
    ((sat > 14) | (val < 100))
).astype(np.uint8)

# Keep largest connected component (the pouch)
labeled, num = ndimage.label(mask)
if num > 0:
    sizes = ndimage.sum(mask, labeled, range(1, num + 1))
    largest = np.argmax(sizes) + 1
    mask = (labeled == largest).astype(np.uint8)

# Fill holes inside the pouch (label area, barcode, etc.)
mask = ndimage.binary_fill_holes(mask).astype(np.uint8)

# Erode to remove white halo / JPEG artifacts at the boundary
kernel = np.ones((3, 3), np.uint8)
mask = cv2.erode(mask, kernel, iterations=1)

# Small dilation to recover natural edge after erosion
mask = cv2.dilate(mask, kernel, iterations=1)

# Final alpha: hard mask with small blur for anti-aliasing
alpha = (mask * 255).astype(np.uint8)
alpha = cv2.GaussianBlur(alpha, (3, 3), 0)

mask_img = Image.fromarray(alpha, mode="L")
img.putalpha(mask_img)

# Crop to content
bbox = img.getbbox()
if bbox:
    left, top, right, bottom = bbox
    # trim a tiny bit from bottom in case shadow survived
    bottom = int(bottom - (bottom - top) * 0.01)
    img = img.crop((left, top, right, bottom))

W2, H2 = img.size
pad = 20
new = Image.new("RGBA", (W2 + pad * 2, H2 + pad * 2), (255, 255, 255, 0))
new.paste(img, (pad, pad), img)

new.save(out, "PNG")
print("Saved clean product to", out, "size", new.size)
