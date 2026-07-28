from PIL import Image, ImageDraw, ImageFilter

src = "E:/Desktop/茶叶/成品_样稿确认/jimeng-2026-07-21-3817-以茶包产品图@图片1为主体，保持牛皮纸自立袋的真实材质、立体形状、挂孔、拉链封口....png"
dst = "E:/Desktop/茶叶/成品_第八轮_主图场景样稿/_assets/product_clean.png"

img = Image.open(src).convert("RGBA")
w, h = img.size

# Tight mask to cut white fringe
mask = Image.new("L", (w, h), 0)
draw = ImageDraw.Draw(mask)

# Tighter rounded rectangle following actual bag edges
x1, y1, x2, y2 = 245, 105, 775, 915
radius = 30
draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=255)

# Feather edges heavily to blend any remaining fringe
mask = mask.filter(ImageFilter.GaussianBlur(3.0))

# Apply
out = Image.new("RGBA", (w, h), (255, 255, 255, 0))
out.paste(img, (0, 0), mask)

# Crop
bbox = out.getbbox()
if bbox:
    margin = 8
    bbox = (
        max(0, bbox[0] - margin),
        max(0, bbox[1] - margin),
        min(w, bbox[2] + margin),
        min(h, bbox[3] + margin),
    )
    out = out.crop(bbox)

out.save(dst)
print("Saved tight-masked product to", dst, "size", out.size)
