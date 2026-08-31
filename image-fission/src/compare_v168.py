"""v168 桌面交付 + 三联对照（原图 / v164禁词版 / v168单词裂变版）"""
from pathlib import Path
from PIL import Image, ImageDraw

DESK = Path("E:/Desktop")

# 1) 桌面交付 v168 单图
v168 = Image.open(r"E:\Desktop\双接口\image-fission\jobs\smoke_v168\v168_skull_5.jpg").convert("RGB")
desk_v168 = DESK / "image-fission-v168-skull_5.jpg"
v168.save(desk_v168, "JPEG", quality=92, optimize=True)
print(f"[desk] v168 {v168.size} {desk_v168.stat().st_size/1024/1024:.2f}MB")

# 2) 三联对照
orig = Image.open(r"E:\Desktop\双接口\image-fission\ComfyUI\input\pinterest_skull_5.jpg").convert("RGB")
v164 = Image.open(r"E:\Desktop\双接口\image-fission\jobs\smoke_v164\v164_skull_5.jpg").convert("RGB")

imgs = [("原图 TRUE/NEVER/DIES", orig), ("v164 禁词干净版", v164), ("v168 单词裂变 BONE/BLOOM/ASH", v168)]
H = 900
gap = 24
parts = []
for label, im in imgs:
    w = int(im.width * H / im.height)
    parts.append((label, im.resize((w, H), Image.LANCZOS)))
total_w = sum(p[1].width for p in parts) + gap * (len(parts) - 1)
canvas = Image.new("RGB", (total_w, H + 56), (24, 24, 24))
d = ImageDraw.Draw(canvas)
x = 0
for label, im in parts:
    canvas.paste(im, (x, 50))
    d.text((x + 10, 12), label, fill=(255, 255, 255))
    x += im.width + gap
out_compare = DESK / "image-fission-v168-skull_5-3way-compare.jpg"
canvas.save(out_compare, "JPEG", quality=90, optimize=True)
print(f"[compare] {out_compare} {out_compare.stat().st_size/1024/1024:.2f}MB")