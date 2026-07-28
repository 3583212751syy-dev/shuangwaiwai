from PIL import Image, ImageFilter

src = "E:/Desktop/茶叶/成品_样稿确认/jimeng-2026-07-21-3817-以茶包产品图@图片1为主体，保持牛皮纸自立袋的真实材质、立体形状、挂孔、拉链封口....png"
dst = "E:/Desktop/茶叶/成品_第八轮_主图场景样稿/_assets/product_clean.png"

img = Image.open(src).convert("RGBA")
w, h = img.size

# Manual crop region covering the product bag
left = 180
top = 50
right = 840
bottom = 960
cropped = img.crop((left, top, right, bottom))
cw, ch = cropped.size

# Make everything outside product silhouette transparent using a simple mask
# Product occupies central vertical strip, leave a small margin
mask = Image.new("L", (cw, ch), 255)
mp = mask.load()

# Top and bottom small rounded margins (keep bag shape)
for y in range(ch):
    for x in range(cw):
        r, g, b, a = cropped.getpixel((x, y))
        # If very bright and near edges, taper alpha
        mx, mn = max(r, g, b), min(r, g, b)
        brightness = (r + g + b) / 3
        # Strongly transparent for pure/near white background
        if brightness > 245 and mx - mn < 12:
            mp[x, y] = 0
        elif brightness > 235 and mx - mn < 8:
            mp[x, y] = 80
        else:
            mp[x, y] = 255

mask = mask.filter(ImageFilter.GaussianBlur(1.0))
out = Image.new("RGBA", (cw, ch), (255, 255, 255, 0))
out.paste(cropped, (0, 0), mask)

out.save(dst)
print("Saved manual crop to", dst, "size", out.size)
