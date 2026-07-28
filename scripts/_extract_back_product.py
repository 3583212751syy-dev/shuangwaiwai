from PIL import Image, ImageFilter, ImageOps
import numpy as np

src = "E:/Desktop/茶叶/产品背面图片.png"
out = "E:/Desktop/茶叶/成品_第九轮_真实站立包装/_assets/product_back_clean.png"

img = Image.open(src).convert("RGBA")
W, H = img.size

# White background removal: background is near pure white
r, g, b, a = img.split()
gray = ImageOps.grayscale(img.convert("RGB"))
arr = np.array(gray)
rgb = np.array(img.convert("RGB"))

# mask: not background
# background: very bright, low saturation
mask = np.ones((H, W), dtype=np.uint8) * 255
for y in range(H):
    for x in range(W):
        R, G, B = rgb[y, x]
        mx = max(R, G, B)
        mn = min(R, G, B)
        # white bg: high brightness, low saturation
        if mx >= 245 and (mx - mn) <= 12:
            mask[y, x] = 0
        # also catch slightly gray white
        elif mx >= 240 and (mx - mn) <= 6:
            mask[y, x] = 0

mask_img = Image.fromarray(mask, mode="L")
# slight blur to soften edges
mask_img = mask_img.filter(ImageFilter.GaussianBlur(0.5))

img.putalpha(mask_img)

# crop to content
bbox = img.getbbox()
if bbox:
    img = img.crop(bbox)

# add small padding
W2, H2 = img.size
pad = 20
new = Image.new("RGBA", (W2 + pad*2, H2 + pad*2), (255, 255, 255, 0))
new.paste(img, (pad, pad), img)

new.save(out, "PNG")
print("Saved clean product to", out)
