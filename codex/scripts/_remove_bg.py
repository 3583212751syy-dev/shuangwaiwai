from PIL import Image, ImageFilter
import os

src = "E:/Desktop/茶叶/成品_样稿确认/jimeng-2026-07-21-3817-以茶包产品图@图片1为主体，保持牛皮纸自立袋的真实材质、立体形状、挂孔、拉链封口....png"
out_dir = "E:/Desktop/茶叶/成品_第七轮_站立图文修正/_assets"
os.makedirs(out_dir, exist_ok=True)

img = Image.open(src).convert("RGBA")
# Scale down to max 1536 while keeping aspect ratio
max_size = 1536
img.thumbnail((max_size, max_size), Image.LANCZOS)

# Create mask using chroma difference for pure white background
w, h = img.size
mask = Image.new("L", (w, h), 255)
px = img.load()
mp = mask.load()

thr_delta = 12
thr_bri = 245

for y in range(h):
    for x in range(w):
        r, g, b, a = px[x, y]
        mx = max(r, g, b)
        mn = min(r, g, b)
        delta = mx - mn
        # White background: low saturation and bright
        if delta < thr_delta and mx >= thr_bri:
            mp[x, y] = 0
        else:
            mp[x, y] = 255

# Slight blur to soften edges
mask = mask.filter(ImageFilter.GaussianBlur(1.0))

# Apply mask
result = Image.new("RGBA", (w, h), (255, 255, 255, 0))
result.paste(img, (0, 0), mask)

out_path = os.path.join(out_dir, "product_standing.png")
result.save(out_path)
print("Saved:", out_path, "size:", result.size)
