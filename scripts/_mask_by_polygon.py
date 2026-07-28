from PIL import Image, ImageDraw, ImageFilter

src = "E:/Desktop/茶叶/成品_样稿确认/jimeng-2026-07-21-3817-以茶包产品图@图片1为主体，保持牛皮纸自立袋的真实材质、立体形状、挂孔、拉链封口....png"
dst = "E:/Desktop/茶叶/成品_第八轮_主图场景样稿/_assets/product_clean.png"

img = Image.open(src).convert("RGBA")
w, h = img.size

# Product silhouette approximated as rounded rectangle based on visual inspection
# Bag body: x 220-800, y 80-940, top corners rounded
mask = Image.new("L", (w, h), 0)
draw = ImageDraw.Draw(mask)

# Draw rounded rectangle for bag body
x1, y1, x2, y2 = 220, 80, 800, 940
radius = 40
draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=255)

# Also remove obvious pure white background in case it leaks inside
mp = mask.load()
px = img.load()
for y in range(h):
    for x in range(w):
        if mp[x, y] == 0:
            continue
        r, g, b, a = px[x, y]
        # If extremely bright inside mask, reduce alpha (background fringe)
        if r > 250 and g > 250 and b > 250:
            mp[x, y] = 0

# Feather edges
mask = mask.filter(ImageFilter.GaussianBlur(2.0))

# Apply mask
out = Image.new("RGBA", (w, h), (255, 255, 255, 0))
out.paste(img, (0, 0), mask)

# Crop to bbox
bbox = out.getbbox()
if bbox:
    margin = 12
    bbox = (
        max(0, bbox[0] - margin),
        max(0, bbox[1] - margin),
        min(w, bbox[2] + margin),
        min(h, bbox[3] + margin),
    )
    out = out.crop(bbox)

out.save(dst)
print("Saved polygon-masked product to", dst, "size", out.size)
