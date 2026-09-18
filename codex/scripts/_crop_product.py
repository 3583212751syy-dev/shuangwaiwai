from PIL import Image, ImageFilter

src = "E:/Desktop/茶叶/成品_第七轮_站立图文修正/_assets/product_standing.png"
dst = "E:/Desktop/茶叶/成品_第八轮_主图场景样稿/_assets/product_clean.png"

img = Image.open(src).convert("RGBA")
w, h = img.size
px = img.load()

# First pass: mark any near-white background as transparent
for y in range(h):
    for x in range(w):
        r, g, b, a = px[x, y]
        mx, mn = max(r, g, b), min(r, g, b)
        delta = mx - mn
        brightness = (r + g + b) / 3
        # Strictly pure/near white background
        if brightness > 248 and delta < 8 and a > 0:
            px[x, y] = (255, 255, 255, 0)

# Get content bounding box
bbox = img.getbbox()
if bbox:
    # Add small margin
    margin = 10
    bbox = (
        max(0, bbox[0] - margin),
        max(0, bbox[1] - margin),
        min(w, bbox[2] + margin),
        min(h, bbox[3] + margin),
    )
    img = img.crop(bbox)

# Remove residual white fringe outside product silhouette by flood-fill-like cleanup
# We keep everything inside the bbox, but clean obvious white edges
w2, h2 = img.size
px2 = img.load()
for y in range(h2):
    for x in range(w2):
        r, g, b, a = px2[x, y]
        if a == 0:
            continue
        mx, mn = max(r, g, b), min(r, g, b)
        delta = mx - mn
        brightness = (r + g + b) / 3
        # If very bright and neutral, likely leftover white background fringe
        if brightness > 245 and delta < 10:
            px2[x, y] = (255, 255, 255, 0)

# Slight alpha blur for soft edges
r, g, b, a = img.split()
a = a.filter(ImageFilter.GaussianBlur(0.5))
img.putalpha(a)

img.save(dst)
print("Saved cropped product to", dst, "size", img.size)
