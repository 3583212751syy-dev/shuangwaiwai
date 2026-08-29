"""v167 桌面交付 + 三联对照（原图 / v164禁词版 / v167单词裂变版）"""
from pathlib import Path
from PIL import Image, ImageDraw

# 桌面交付
DESK = Path("E:/Desktop")
v167 = Image.open(r"E:\Desktop\双接口\image-fission\jobs\smoke_v167\v167_metal_6.jpg").convert("RGB")
v167.save(DESK / "image-fission-v167-metal_6.jpg", "JPEG", quality=92, optimize=True)
print(f"[desk] v167 桌面 {v167.size} {(DESK / 'image-fission-v167-metal_6.jpg').stat().st_size/1024/1024:.2f}MB")

# 三联对照
orig = Image.open(r"E:\Desktop\双接口\image-fission\ComfyUI\input\pinterest_metal_6.jpg").convert("RGB")
v164 = Image.open(r"E:\Desktop\双接口\image-fission\jobs\smoke_v164\v164_metal_6.jpg").convert("RGB")

imgs = [("原图 (MRCHGSR logo)", orig), ("v164 禁词版 (无顶部字)", v164), ("v167 单词裂变 (AI 拼写)", v167)]
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
out_compare = DESK / "image-fission-v167-metal_6-3way-compare.jpg"
canvas.save(out_compare, "JPEG", quality=90, optimize=True)
print(f"[compare] {out_compare} {(out_compare.stat().st_size)/1024/1024:.2f}MB")
