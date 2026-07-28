from PIL import Image, ImageFilter
import sys

src = "E:/Desktop/茶叶/成品_样稿确认/jimeng-2026-07-21-3817-以茶包产品图@图片1为主体，保持牛皮纸自立袋的真实材质、立体形状、挂孔、拉链封口....png"
dst = "E:/Desktop/茶叶/成品_第八轮_主图场景样稿/_assets/product_clean.png"

img = Image.open(src).convert("RGBA")
W, H = img.size

# Build mask: background near-white becomes transparent
px = img.load()
mask = Image.new("L", (W, H), 255)
mp = mask.load()

# Threshold for background: very light and low saturation
for y in range(H):
    for x in range(W):
        r, g, b, a = px[x, y]
        mx = max(r, g, b)
        mn = min(r, g, b)
        delta = mx - mn
        # background: bright, low saturation, or white
        if mx >= 245 and delta <= 12:
            mp[x, y] = 0
        else:
            mp[x, y] = 255

# Flood fill from corners to remove background
from PIL import ImageDraw
fill = Image.new("L", (W, H), 0)
fd = fill.load()
stack = [(0,0), (W-1,0), (0,H-1), (W-1,H-1)]
visited = set(stack)
thresh = 20
bg_px = px
while stack:
    x, y = stack.pop()
    if mp[x, y] == 0:
        fd[x, y] = 255
        for nx, ny in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
            if 0 <= nx < W and 0 <= ny < H and (nx,ny) not in visited:
                r,g,b,a = bg_px[nx,ny]
                mx = max(r,g,b); mn = min(r,g,b)
                if mx >= 235 and (mx-mn) <= thresh:
                    visited.add((nx,ny))
                    stack.append((nx,ny))

# Combine: keep only non-background
final_mask = Image.new("L", (W, H), 0)
fm = final_mask.load()
fp = fill.load()
for y in range(H):
    for x in range(W):
        if mp[x,y] == 255 and fp[x,y] == 0:
            fm[x,y] = 255

# Small blur to soften edges
final_mask = final_mask.filter(ImageFilter.GaussianBlur(0.7))

# Apply mask
out = Image.new("RGBA", (W, H), (0,0,0,0))
out.paste(img, (0,0), final_mask)

# Crop to content bbox
bbox = out.getbbox()
if bbox:
    out = out.crop(bbox)

out.save(dst)
print("Saved", dst, "size", out.size)
