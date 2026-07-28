from PIL import Image, ImageFilter
import os

src = "E:/Desktop/茶叶/成品_样稿确认/jimeng-2026-07-21-3817-以茶包产品图@图片1为主体，保持牛皮纸自立袋的真实材质、立体形状、挂孔、拉链封口....png"
out_dir = "E:/Desktop/茶叶/成品_第七轮_站立图文修正/_assets"
os.makedirs(out_dir, exist_ok=True)

img = Image.open(src).convert("RGBA")
max_size = 1536
img.thumbnail((max_size, max_size), Image.LANCZOS)

w, h = img.size
mask = Image.new("L", (w, h), 255)
px = img.load()
mp = mask.load()

for y in range(h):
    for x in range(w):
        r, g, b, a = px[x, y]
        mx = max(r, g, b)
        mn = min(r, g, b)
        delta = mx - mn
        # Pure white / near white background -> transparent
        if mx >= 252 or (delta < 18 and mx >= 230):
            mp[x, y] = 0
        else:
            mp[x, y] = 255

# Erode slightly to remove edge fringes
mask = mask.filter(ImageFilter.MinFilter(3))
# Then very small blur for anti-alias
mask = mask.filter(ImageFilter.GaussianBlur(0.5))

result = Image.new("RGBA", (w, h), (255, 255, 255, 0))
result.paste(img, (0, 0), mask)

out_path = os.path.join(out_dir, "product_standing.png")
result.save(out_path)
print("Saved:", out_path, "size:", result.size)
