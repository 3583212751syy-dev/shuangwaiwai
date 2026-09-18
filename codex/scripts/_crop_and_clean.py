from PIL import Image, ImageFilter

src = "E:/Desktop/茶叶/成品_样稿确认/jimeng-2026-07-21-3817-以茶包产品图@图片1为主体，保持牛皮纸自立袋的真实材质、立体形状、挂孔、拉链封口....png"
dst = "E:/Desktop/茶叶/成品_第八轮_主图场景样稿/_assets/product_clean.png"

img = Image.open(src).convert("RGBA")
w, h = img.size

# Hard crop to remove most white border
left = int(w * 0.10)
top = int(h * 0.02)
right = int(w * 0.90)
bottom = int(h * 0.98)
img = img.crop((left, top, right, bottom))
w, h = img.size
px = img.load()

# Conservative background removal: only near-white pixels become transparent
for y in range(h):
    for x in range(w):
        r, g, b, a = px[x, y]
        mx, mn = max(r, g, b), min(r, g, b)
        delta = mx - mn
        brightness = (r + g + b) / 3
        if brightness > 248 and delta < 10 and a > 0:
            px[x, y] = (255, 255, 255, 0)

# Alpha blur for smooth edges
r, g, b, a = img.split()
a = a.filter(ImageFilter.GaussianBlur(0.6))
img.putalpha(a)

img.save(dst)
print("Saved cleaned product to", dst, "size", img.size)
