from PIL import Image, ImageFilter
import sys

src = "E:/Desktop/茶叶/成品_样稿确认/jimeng-2026-07-21-3817-以茶包产品图@图片1为主体，保持牛皮纸自立袋的真实材质、立体形状、挂孔、拉链封口....png"
dst = "E:/Desktop/茶叶/成品_第八轮_主图场景样稿/_assets/product_clean.png"

img = Image.open(src).convert("RGBA")
W, H = img.size
px = img.load()

# Strict background: near-white, very low saturation
mask = Image.new("L", (W, H), 0)
mp = mask.load()
for y in range(H):
    for x in range(W):
        r, g, b, a = px[x, y]
        mx = max(r, g, b)
        mn = min(r, g, b)
        if mx >= 190 and (mx - mn) <= 22:
            mp[x, y] = 255
        else:
            mp[x, y] = 0

# Flood fill from edges on strict mask
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

while stack:
    x, y = stack.pop()
    if mp[x, y] == 255:
        fd[x, y] = 255
        for nx, ny in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
            if 0 <= nx < W and 0 <= ny < H and (nx,ny) not in visited and mp[nx,ny] == 255:
                visited.add((nx,ny))
                stack.append((nx,ny))

# Dilate background slightly to catch edge shadows, then erode back
for _ in range(3):
    fill = fill.filter(ImageFilter.MaxFilter(3))
for _ in range(1):
    fill = fill.filter(ImageFilter.MinFilter(3))

# Product mask = not background
prod = Image.new("L", (W, H), 0)
pp = prod.load()
fp = fill.load()
for y in range(H):
    for x in range(W):
        if fp[x,y] == 0:
            pp[x,y] = 255

# Remove small isolated noise by connected components
comp_id = Image.new("L", (W, H), 0)
cp = comp_id.load()
comp_areas = {}
curr_id = 1
for y in range(H):
    for x in range(W):
        if pp[x,y] == 255 and cp[x,y] == 0:
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

max_id = max(comp_areas, key=comp_areas.get)
print("Components:", curr_id-1, "areas:", sorted(comp_areas.values(), reverse=True)[:5])
for y in range(H):
    for x in range(W):
        if cp[x,y] != max_id:
            pp[x,y] = 0

# Small edge blur for antialiasing
prod = prod.filter(ImageFilter.GaussianBlur(0.5))

out = Image.new("RGBA", (W, H), (0,0,0,0))
out.paste(img, (0,0), prod)

bbox = out.getbbox()
if bbox:
    out = out.crop(bbox)

out.save(dst)
print("Saved", dst, "size", out.size)
