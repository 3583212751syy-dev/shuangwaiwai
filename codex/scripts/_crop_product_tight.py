from PIL import Image

src = "E:/Desktop/茶叶/成品_样稿确认/jimeng-2026-07-21-3817-以茶包产品图@图片1为主体，保持牛皮纸自立袋的真实材质、立体形状、挂孔、拉链封口....png"
dst = "E:/Desktop/茶叶/成品_第八轮_主图场景样稿/_assets/product_clean.png"

img = Image.open(src).convert("RGBA")
w, h = img.size
px = img.load()

# Make obvious pure white background transparent
for y in range(h):
    for x in range(w):
        r, g, b, a = px[x, y]
        if r > 250 and g > 250 and b > 250 and a > 0:
            px[x, y] = (255, 255, 255, 0)

# Crop to content bbox
bbox = img.getbbox()
if bbox:
    margin = 8
    bbox = (
        max(0, bbox[0] - margin),
        max(0, bbox[1] - margin),
        min(w, bbox[2] + margin),
        min(h, bbox[3] + margin),
    )
    img = img.crop(bbox)

img.save(dst)
print("Saved cropped product to", dst, "size", img.size)
