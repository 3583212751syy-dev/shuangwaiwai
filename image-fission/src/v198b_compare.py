"""v198_compare.py — v198b vs 原图 vs v193 三列拼图对照"""
from PIL import Image, ImageDraw, ImageFont
import os

OUT = "E:/Desktop/双接口/image-fission/jobs/smoke_v198b/_6up_compare_v198b.jpg"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

paths = [
    ("原图（迷彩底）", "E:/Desktop/双接口/image-fission/ComfyUI/input/test_5784eab326634d17573b469e91cdc565.jpg"),
    ("v193 bat_logo（紫底，违背景色）", "E:/Desktop/双接口/image-fission/jobs/smoke_v193/v193_bat_logo.jpg"),
    ("v198b bat_logo（迷彩+清晰字，最终）", "E:/Desktop/双接口/image-fission/jobs/smoke_v198b/v198b_bat_logo_burned.jpg"),
]

cells = []
for label, p in paths:
    im = Image.open(p).convert("RGB")
    im.thumbnail((1024, 1400), Image.LANCZOS)
    cells.append((label, im))

label_h = 60
gap = 30
W = sum(c[1].width for c in cells) + gap * (len(cells) + 1)
H = max(c[1].height for c in cells) + label_h + gap * 2

canvas = Image.new("RGB", (W, H), (235, 232, 226))
font = ImageFont.truetype("E:/Desktop/双接口/image-fission/fonts/PirataOne-Regular.ttf", 36)
draw = ImageDraw.Draw(canvas)

x = gap
for label, im in cells:
    canvas.paste(im, (x, label_h + gap))
    bb = draw.textbbox((0, 0), label, font=font)
    tw = bb[2] - bb[0]
    draw.text((x + (im.width - tw) // 2, gap), label, font=font, fill=(40, 20, 60))
    x += im.width + gap

canvas.save(OUT, quality=92)
print(f"saved {OUT}")
print(f"size: {W}x{H}")