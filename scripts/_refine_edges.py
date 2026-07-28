from PIL import Image, ImageFilter

src = "E:/Desktop/茶叶/成品_第八轮_主图场景样稿/_assets/product_clean.png"
dst = "E:/Desktop/茶叶/成品_第八轮_主图场景样稿/_assets/product_clean.png"

img = Image.open(src).convert("RGBA")
w, h = img.size
px = img.load()

# For edge pixels (alpha between 20 and 250), reduce alpha if they are bright
# This removes white fringe while preserving solid product
for y in range(h):
    for x in range(w):
        r, g, b, a = px[x, y]
        if a < 10 or a > 250:
            continue
        brightness = (r + g + b) / 3
        # Bright fringe -> reduce alpha
        if brightness > 200:
            new_a = int(a * (1 - (brightness - 200) / 80))
            new_a = max(0, min(255, new_a))
            px[x, y] = (r, g, b, new_a)

# Also for solid edge pixels near fully opaque, if very bright reduce slightly
for y in range(h):
    for x in range(w):
        r, g, b, a = px[x, y]
        if a < 250:
            continue
        brightness = (r + g + b) / 3
        mx, mn = max(r, g, b), min(r, g, b)
        # Very bright solid pixels likely white background inside mask
        if brightness > 245 and mx - mn < 10:
            px[x, y] = (r, g, b, 0)

# Alpha blur for smooth transition
r, g, b, a = img.split()
a = a.filter(ImageFilter.GaussianBlur(0.5))
img.putalpha(a)

img.save(dst)
print("Refined edges saved to", dst)
