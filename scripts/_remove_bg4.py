from PIL import Image, ImageDraw, ImageFilter
from PIL.ImageDraw import floodfill
import os

src = "E:/Desktop/茶叶/成品_样稿确认/jimeng-2026-07-21-3817-以茶包产品图@图片1为主体，保持牛皮纸自立袋的真实材质、立体形状、挂孔、拉链封口....png"
out_dir = "E:/Desktop/茶叶/成品_第七轮_站立图文修正/_assets"
os.makedirs(out_dir, exist_ok=True)

img = Image.open(src).convert("RGBA")
max_size = 1536
img.thumbnail((max_size, max_size), Image.LANCZOS)

w, h = img.size
bg = Image.new("L", (w, h), 0)
px = img.load()
mp = bg.load()

for y in range(h):
    for x in range(w):
        r, g, b, a = px[x, y]
        mx = max(r, g, b)
        mn = min(r, g, b)
        delta = mx - mn
        if mx >= 248 and delta < 15:
            mp[x, y] = 255
        else:
            mp[x, y] = 0

# Flood fill connected background from corners
corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
for cx, cy in corners:
    if bg.getpixel((cx, cy)) == 255:
        floodfill(bg, (cx, cy), 128)

mask = bg.point(lambda v: 0 if v == 128 else 255)
mask = mask.filter(ImageFilter.GaussianBlur(0.5))

result = Image.new("RGBA", (w, h), (255, 255, 255, 0))
result.paste(img, (0, 0), mask)

out_path = os.path.join(out_dir, "product_standing.png")
result.save(out_path)
print("Saved:", out_path, "size:", result.size)
