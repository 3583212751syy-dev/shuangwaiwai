from PIL import Image, ImageFilter

src = "E:/Desktop/茶叶/产品图.png"
dst = "E:/Desktop/茶叶/成品_第八轮_主图场景样稿/_assets/product_clean.png"

img = Image.open(src).convert("RGBA")
w, h = img.size

# Build mask: treat near-white / light gray backgrounds as transparent
mask = Image.new("L", (w, h), 0)
mp = mask.load()
px = img.load()

thr_delta = 18
thr_bri = 235

for y in range(h):
    for x in range(w):
        r, g, b, a = px[x, y]
        mx, mn = max(r, g, b), min(r, g, b)
        delta = mx - mn
        # If near neutral and bright, make transparent
        if delta < thr_delta and mx >= thr_bri:
            mp[x, y] = 0
        else:
            mp[x, y] = 255

# Slight blur for soft edges
mask = mask.filter(ImageFilter.GaussianBlur(1.0))

# Apply mask
out = Image.new("RGBA", (w, h), (255, 255, 255, 0))
out.paste(img, (0, 0), mask)

# Crop to content
bbox = out.getbbox()
if bbox:
    out = out.crop(bbox)

# Add small padding
pad = 20
new = Image.new("RGBA", (out.width + pad*2, out.height + pad*2), (255, 255, 255, 0))
new.paste(out, (pad, pad))

new.save(dst)
print("Saved clean product to", dst)
