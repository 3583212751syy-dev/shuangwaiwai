from PIL import Image, ImageFilter

src = "E:/Desktop/茶叶/成品_第七轮_站立图文修正/_assets/product_standing.png"
dst = "E:/Desktop/茶叶/成品_第八轮_主图场景样稿/_assets/product_clean.png"

img = Image.open(src).convert("RGBA")
w, h = img.size
px = img.load()

# Remove near-white background pixels while preserving product and soft shadow
for y in range(h):
    for x in range(w):
        r, g, b, a = px[x, y]
        mx, mn = max(r, g, b), min(r, g, b)
        delta = mx - mn
        brightness = (r + g + b) / 3
        # If pixel is bright, near-neutral, and not already transparent -> make transparent
        if brightness > 238 and delta < 20 and a > 0:
            px[x, y] = (255, 255, 255, 0)

# Also remove semi-opaque white fringe
for y in range(h):
    for x in range(w):
        r, g, b, a = px[x, y]
        mx, mn = max(r, g, b), min(r, g, b)
        delta = mx - mn
        brightness = (r + g + b) / 3
        if brightness > 245 and delta < 15:
            px[x, y] = (255, 255, 255, 0)

# Split and blur alpha for clean edges
r, g, b, a = img.split()
a = a.filter(ImageFilter.GaussianBlur(0.5))
img.putalpha(a)

img.save(dst)
print("Saved cleaned product to", dst)
