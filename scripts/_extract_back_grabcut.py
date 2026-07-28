from PIL import Image, ImageFilter, ImageOps
import numpy as np

src = "E:/Desktop/茶叶/产品背面图片.png"
out = "E:/Desktop/茶叶/成品_第九轮_真实站立包装/_assets/product_back_clean.png"

img = Image.open(src).convert("RGBA")
W, H = img.size
rgb = np.array(img.convert("RGB"))

# Convert to HSV manually or use simple color distance
# White background: high value, low saturation
# Calculate saturation-like metric
maxc = rgb.max(axis=2).astype(float)
minc = rgb.min(axis=2).astype(float)
delta = maxc - minc
saturation = np.where(maxc > 0, delta / maxc, 0)

# Background mask: bright AND low saturation
bg_mask = (maxc >= 230) & (saturation <= 0.12)

# Also include near-white low saturation even if slightly darker
bg_mask2 = (maxc >= 210) & (saturation <= 0.06)
bg_mask = bg_mask | bg_mask2

# Initial mask: background=0, unknown=2, probable foreground=3
# Use morphological operations to clean
mask = np.where(bg_mask, 0, 2).astype('uint8')

# Erode background mask to avoid eating edges
from scipy import ndimage
bg_mask = ndimage.binary_erosion(bg_mask, iterations=1)
mask = np.where(bg_mask, 0, 2).astype('uint8')

# Mark center as probable foreground
mask[H//3:2*H//3, W//3:2*W//3] = 3

try:
    import cv2
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    mask2, bgdModel, fgdModel = cv2.grabCut(rgb, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    final_mask = np.where((mask2 == 2) | (mask2 == 0), 0, 255).astype('uint8')
    final_mask = ndimage.binary_closing(final_mask > 0, iterations=2).astype(np.uint8) * 255
    final_mask = ndimage.binary_opening(final_mask > 0, iterations=1).astype(np.uint8) * 255
    mask_img = Image.fromarray(final_mask, mode="L").filter(ImageFilter.GaussianBlur(0.7))
except Exception as e:
    print("grabCut failed:", e)
    # fallback
    mask_img = Image.fromarray((~bg_mask).astype(np.uint8) * 255, mode="L")

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
