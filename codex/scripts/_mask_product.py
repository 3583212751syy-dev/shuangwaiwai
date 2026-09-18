from PIL import Image, ImageFilter

src = "E:/Desktop/茶叶/成品_样稿确认/jimeng-2026-07-21-3817-以茶包产品图@图片1为主体，保持牛皮纸自立袋的真实材质、立体形状、挂孔、拉链封口....png"
dst = "E:/Desktop/茶叶/成品_第八轮_主图场景样稿/_assets/product_clean.png"

img = Image.open(src).convert("RGBA")
w, h = img.size
px = img.load()

# Initial mask: white background = 0, content = 255
mask = Image.new("L", (w, h), 0)
mp = mask.load()
for y in range(h):
    for x in range(w):
        r, g, b, a = px[x, y]
        mx, mn = max(r, g, b), min(r, g, b)
        delta = mx - mn
        brightness = (r + g + b) / 3
        # Treat near-white as background
        if brightness > 245 and delta < 12:
            mp[x, y] = 0
        else:
            mp[x, y] = 255

# Keep only the largest connected component (the product)
# Simple flood-fill to find largest component
visited = [[False]*w for _ in range(h)]
best_comp = []
for y in range(h):
    for x in range(w):
        if mp[x, y] == 255 and not visited[y][x]:
            stack = [(x, y)]
            comp = []
            visited[y][x] = True
            while stack:
                cx, cy = stack.pop()
                comp.append((cx, cy))
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = cx+dx, cy+dy
                    if 0 <= nx < w and 0 <= ny < h and mp[nx, ny] == 255 and not visited[ny][nx]:
                        visited[ny][nx] = True
                        stack.append((nx, ny))
            if len(comp) > len(best_comp):
                best_comp = comp

# Reset mask to keep only largest component
mask = Image.new("L", (w, h), 0)
mp = mask.load()
for x, y in best_comp:
    mp[x, y] = 255

# Dilate slightly to include anti-aliased edges
mask = mask.filter(ImageFilter.MaxFilter(3))
mask = mask.filter(ImageFilter.GaussianBlur(1.0))

# Apply mask
out = Image.new("RGBA", (w, h), (255, 255, 255, 0))
out.paste(img, (0, 0), mask)

# Crop to content
bbox = out.getbbox()
if bbox:
    margin = 10
    bbox = (
        max(0, bbox[0] - margin),
        max(0, bbox[1] - margin),
        min(w, bbox[2] + margin),
        min(h, bbox[3] + margin),
    )
    out = out.crop(bbox)

out.save(dst)
print("Saved cleaned product to", dst, "size", out.size)
