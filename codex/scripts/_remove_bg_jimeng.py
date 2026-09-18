from PIL import Image, ImageFilter

src = "E:/Desktop/茶叶/成品_样稿确认/jimeng-2026-07-21-3817-以茶包产品图@图片1为主体，保持牛皮纸自立袋的真实材质、立体形状、挂孔、拉链封口....png"
dst = "E:/Desktop/茶叶/成品_第八轮_主图场景样稿/_assets/product_clean.png"

img = Image.open(src).convert("RGBA")
w, h = img.size
px = img.load()

# Very conservative: only pure/near-pure white becomes transparent
# Label is off-white/cream, so it stays
for y in range(h):
    for x in range(w):
        r, g, b, a = px[x, y]
        mx, mn = max(r, g, b), min(r, g, b)
        delta = mx - mn
        brightness = (r + g + b) / 3
        # Pure white background
        if brightness > 252 and delta < 6 and a > 0:
            px[x, y] = (255, 255, 255, 0)

# Slight alpha blur for anti-aliased edges
r, g, b, a = img.split()
a = a.filter(ImageFilter.GaussianBlur(0.5))
img.putalpha(a)

# Crop to content
bbox = img.getbbox()
if bbox:
    margin = 10
    bbox = (
        max(0, bbox[0] - margin),
        max(0, bbox[1] - margin),
        min(w, bbox[2] + margin),
        min(h, bbox[3] + margin),
    )
    img = img.crop(bbox)

img.save(dst)
print("Saved clean product to", dst, "size", img.size)
