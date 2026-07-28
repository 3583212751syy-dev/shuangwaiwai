from PIL import Image, ImageFilter, ImageMorph
import sys

src = "E:/Desktop/茶叶/成品_样稿确认/jimeng-2026-07-21-3817-以茶包产品图@图片1为主体，保持牛皮纸自立袋的真实材质、立体形状、挂孔、拉链封口....png"
dst = "E:/Desktop/茶叶/成品_第八轮_主图场景样稿/_assets/product_clean.png"

img = Image.open(src).convert("RGBA")
W, H = img.size
px = img.load()

# Step 1: initial background mask (bright and low saturation)
mask = Image.new("L", (W, H), 0)
mp = mask.load()
for y in range(H):
    for x in range(W):
        r, g, b, a = px[x, y]
        mx = max(r, g, b)
        mn = min(r, g, b)
        delta = mx - mn
        # background: light and near-gray
        if mx >= 200 and delta <= 35:
            mp[x, y] = 255
        else:
            mp[x, y] = 0

# Step 2: flood fill from all four edges
fill = Image.new("L", (W, H), 0)
fd = fill.load()
stack = []
for x in range(W):
    stack.append((x, 0))
    stack.append((x, H-1))
for y in range(H):
    stack.append((0, y))
    stack.append((W-1, y))
visited = set(stack)
thresh_bright = 180
thresh_delta = 50

while stack:
    x, y = stack.pop()
    r, g, b, a = px[x, y]
    mx = max(r, g, b); mn = min(r, g, b)
    if mx >= thresh_bright and (mx-mn) <= thresh_delta:
        fd[x, y] = 255
        for nx, ny in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
            if 0 <= nx < W and 0 <= ny < H and (nx,ny) not in visited:
                visited.add((nx,ny))
                stack.append((nx,ny))

# Step 3: combine masks - consider pixel background if either says so
bg = Image.new("L", (W, H), 0)
bg_p = bg.load()
fp = fill.load()
for y in range(H):
    for x in range(W):
        if mp[x,y] == 255 or fp[x,y] == 255:
            bg_p[x,y] = 255

# Step 4: close holes / remove noise with morphological operations
# Dilate to fill small gaps, then erode to restore shape
for _ in range(3):
    bg = bg.filter(ImageFilter.MaxFilter(3))
for _ in range(3):
    bg = bg.filter(ImageFilter.MinFilter(3))

# Invert to get product mask
prod = Image.new("L", (W, H), 0)
pp = prod.load()
bp = bg.load()
for y in range(H):
    for x in range(W):
        if bp[x,y] == 0:
            pp[x,y] = 255

# Remove small noise: connected components filtering by area
from PIL import Image as PILImage
# Use simple flood fill to find components
comp_id = Image.new("L", (W, H), 0)
cp = comp_id.load()
comp_areas = {}
curr_id = 1
for y in range(H):
    for x in range(W):
        if pp[x,y] == 255 and cp[x,y] == 0:
            # BFS
            area = 0
            stack2 = [(x,y)]
            cp[x,y] = curr_id
            while stack2:
                cx, cy = stack2.pop()
                area += 1
                for nx, ny in ((cx+1,cy),(cx-1,cy),(cx,cy+1),(cx,cy-1)):
                    if 0 <= nx < W and 0 <= ny < H and pp[nx,ny] == 255 and cp[nx,ny] == 0:
                        cp[nx,ny] = curr_id
                        stack2.append((nx,ny))
            comp_areas[curr_id] = area
            curr_id += 1

# Keep only largest component
max_id = max(comp_areas, key=comp_areas.get)
print("Components:", curr_id-1, "areas:", sorted(comp_areas.values(), reverse=True)[:5])
for y in range(H):
    for x in range(W):
        if cp[x,y] != max_id:
            pp[x,y] = 0

# Slight blur to antialias edges
prod = prod.filter(ImageFilter.GaussianBlur(0.6))

# Apply
out = Image.new("RGBA", (W, H), (0,0,0,0))
out.paste(img, (0,0), prod)

bbox = out.getbbox()
if bbox:
    out = out.crop(bbox)

out.save(dst)
print("Saved", dst, "size", out.size)
